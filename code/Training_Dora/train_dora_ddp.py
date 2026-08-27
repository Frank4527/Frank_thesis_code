"""
train_dora_ddp.py — DDP (multi-GPU) version of train_dora.py.

Same training as train_dora.py, but runs one process per GPU via torchrun, so the
work is split across GPUs and synced every step (DistributedDataParallel). Produces
a normal DoRA adapter, identical in format to the single-GPU one.

Launch:
  torchrun --nproc_per_node=4 Training_Dora/train_dora_ddp.py --model ... --task extraction ...

DDP-safety notes (vs the single-GPU file):
  - each process uses its own GPU: cuda:LOCAL_RANK
  - the dev eval / adapter save / prints happen on RANK 0 only
  - the early-stop decision is BROADCAST from rank 0 so all ranks stop together
    (otherwise ranks diverge and the run deadlocks)
  - effective batch = batch_size * num_gpus * grad_accum (pass grad-accum to match single-GPU)
"""
import argparse
import json
import os

# Must precede any tokenizer/torch import (Rust tokenizer thread pool + fork deadlock).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist


def _is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["qa", "extraction"], default="qa")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--eval-every", type=int, default=50, help="optimizer steps")
    ap.add_argument("--dev-limit", type=int, default=200)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dev-max-new-tokens", type=int, default=1024,
                    help="cap dev generation (early-stop only; keeps evals fast)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=50,
                    help="checkpoint interval in optimizer steps")
    ap.add_argument("--save-only-model", action="store_true",
                    help="checkpoints hold the adapter only, not optimizer state "
                         "(~3x smaller; a run saved this way cannot be resumed)")
    args = ap.parse_args()

    # each DDP process pins to its own GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    import random
    import numpy as np
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
    from transformers.trainer_callback import TrainerCallback
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from Training_Dora.qa_data import DocVQACollator, load_manifest
    from Identification_and_Reversion.generate_and_score import evaluate
    from Metrics.document_extraction import ExtractionCollator, load_documents, evaluate_extraction

    if _is_main():
        os.makedirs(args.out, exist_ok=True)

    if _is_main():
        print(f"loading {args.model} on {device} (world size {os.environ.get('WORLD_SIZE', 1)}) ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device)

    if args.task == "extraction":
        all_train = load_documents(args.train)
        if args.dev:
            dev_rows = load_documents(args.dev)[:args.dev_limit]
            train_rows = all_train[:args.limit_train] if args.limit_train else all_train
        else:
            dev_rows = all_train[-args.dev_limit:]
            train_rows = all_train[:-args.dev_limit]
            if args.limit_train:
                train_rows = train_rows[:args.limit_train]
        collator = ExtractionCollator(processor, args.max_pixels)

        # dev generation cap: size to this split's longest target, but hard-cap low so
        # early-stop evals stay fast (final test uses the full cap separately).
        try:
            _max_chars = max(len(json.dumps(r.get("gt_parse", {}), ensure_ascii=False))
                             for r in dev_rows)
            dev_gen_cap = max(512, min(args.dev_max_new_tokens, int(_max_chars / 2.6) + 384))
        except Exception:
            dev_gen_cap = args.dev_max_new_tokens
        if _is_main():
            print(f"dev generation cap = {dev_gen_cap} tokens (dev {len(dev_rows)} docs)", flush=True)

        def dev_metric():
            m = evaluate_extraction(model, processor, dev_rows, batch_size=2,
                                    max_new_tokens=dev_gen_cap,
                                    max_pixels=args.max_pixels, device=device)
            return m["field_f1"], {"field_f1": m["field_f1"], "nted": m["nted"],
                                   "value_f1": m["value_f1"]}
        metric_name = "field_f1"
    else:
        train_rows = load_manifest(args.train, args.limit_train)
        dev_rows = load_manifest(args.dev, args.dev_limit)
        collator = DocVQACollator(processor, args.max_pixels)

        def dev_metric():
            m, _ = evaluate(model, processor, dev_rows, batch_size=8,
                            max_pixels=args.max_pixels, device=device)
            return m["anls"], {"anls": m["anls"], "exact_match": m["exact_match"]}
        metric_name = "anls"
    if _is_main():
        print(f"task={args.task}  train {len(train_rows)} | dev {len(dev_rows)}", flush=True)

    cfg = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_dora=True, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    if _is_main():
        model.print_trainable_parameters()

    class DevEval(TrainerCallback):
        """Generative dev metric + early stopping, DDP-safe: eval/save on rank 0,
        broadcast the stop decision so all ranks halt together."""

        def __init__(self):
            self.best = -1.0
            self.bad = 0
            self.history = []

        def on_step_end(self, cfg_, state, control, **kw):
            if state.global_step == 0 or state.global_step % args.eval_every:
                return
            stop = 0
            if _is_main():
                score, extra = dev_metric()
                model.train()
                self.history.append({"step": state.global_step, **extra})
                print(f"  [dev] step {state.global_step}  "
                      + "  ".join(f"{k} {v:.4f}" for k, v in extra.items()), flush=True)
                if score > self.best:
                    self.best = score
                    self.bad = 0
                    model.save_pretrained(os.path.join(args.out, "adapter_best"))
                    print(f"  [dev] new best {metric_name} {score:.4f} -> adapter_best", flush=True)
                else:
                    self.bad += 1
                    if self.bad >= args.patience:
                        print(f"  [dev] no improvement in {self.bad} evals -> stop", flush=True)
                        stop = 1
            # all ranks agree on whether to stop (prevents deadlock)
            if dist.is_available() and dist.is_initialized():
                t = torch.tensor([stop], device=device)
                dist.broadcast(t, src=0)
                stop = int(t.item())
            if stop:
                control.should_training_stop = True
            return control

    dev_cb = DevEval()

    sft = SFTConfig(
        output_dir=os.path.join(args.out, "hf"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        seed=args.seed,
        data_seed=args.seed,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="steps",     # periodic full checkpoints so a crash can RESUME
        save_steps=args.save_steps,
        save_only_model=args.save_only_model,
        # keep ALL checkpoints: they are the early-stopping baseline, which is
        # the direct alternative to selective reversion (both trade new-task
        # performance for old-task retention). Affects disk only, never training.
        save_total_limit=None,
        bf16=True,
        loss_type="nll",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.workers > 0,
        ddp_find_unused_parameters=False,
        ddp_timeout=7200,   # 2h: tolerate the (slow) rank-0 dev eval without NCCL watchdog abort
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_rows,
        data_collator=collator,
        processing_class=processor,
        # In-loop generative eval is OPT-IN, enabled only when --eval-every is set to a
        # real interval. Fixed-budget runs pass a sentinel (999999999) and therefore get
        # no callback at all -- byte-identical behaviour to before this was wired up.
        # It was previously disabled outright after an OOM at 8B; ddp_timeout=7200 below
        # exists to cover the slow rank-0 eval, and headroom is now ~55GB of 80, so it is
        # re-enabled behind the flag and smoke-tested before any long run.
        callbacks=[dev_cb] if args.eval_every < 1_000_000 else [],
    )

    # auto-resume from the latest checkpoint if one exists (crash-resilient):
    # relaunching the SAME command picks up where it left off.
    from transformers.trainer_utils import get_last_checkpoint
    resume = None
    hf_dir = os.path.join(args.out, "hf")
    if os.path.isdir(hf_dir):
        resume = get_last_checkpoint(hf_dir)
    if _is_main():
        print(f"\ntraining ... (resume_from_checkpoint={resume})", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    loss_history = [{"step": h["step"], "loss": h["loss"]}
                    for h in trainer.state.log_history if "loss" in h]
    losses = [h["loss"] for h in loss_history]

    if _is_main():
        model.save_pretrained(os.path.join(args.out, "adapter_last"))
        summary = {
            "model": args.model, "train_manifest": args.train, "n_train": len(train_rows),
            "rank": args.rank, "lora_alpha": args.lora_alpha, "lr": args.lr,
            "epochs": args.epochs, "seed": args.seed,
            "world_size": int(os.environ.get("WORLD_SIZE", 1)),
            "effective_batch": args.batch_size * int(os.environ.get("WORLD_SIZE", 1)) * args.grad_accum,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "task": args.task, "dev_metric": metric_name, "best_dev_metric": dev_cb.best,
            "loss_history": loss_history, "dev_history": dev_cb.history,
        }
        with open(os.path.join(args.out, "train_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nbest dev {metric_name} {dev_cb.best:.4f}")
        _best = os.path.join(args.out, "adapter_best")
        print(f"adapters -> {args.out}/adapter_last" + (f" (+ adapter_best)" if os.path.isdir(_best) else ""))

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
