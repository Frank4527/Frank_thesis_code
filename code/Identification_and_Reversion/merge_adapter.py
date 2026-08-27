"""Merge a trained DoRA adapter into the base weights -> a full checkpoint.
Used to build the stage-1 language model that becomes the base for stage-2
sequential fine-tuning (and the reversion anchor)."""
import os, argparse, shutil, torch
os.environ.setdefault("EXTRACT_INSTRUCTION", "Extract all fields from this document as JSON.")
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from Identification_and_Reversion.dora_merge import DoRAAdapter, VERSION_FNS

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)      # base Qwen3-VL
ap.add_argument("--adapter", required=True)    # stage-1 adapter_best dir
ap.add_argument("--out", required=True)        # merged checkpoint dir
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

proc = AutoProcessor.from_pretrained(args.model)
model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device)
model.eval()
adapter = DoRAAdapter.load(args.adapter, model, args.device)
adapter.apply(model, VERSION_FNS["full"])      # write fully-trained weights in place
os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=True)
proc.save_pretrained(args.out)
# proc.save_pretrained misses the image/video preprocessor + vocab/merges for Qwen3-VL;
# copy any non-weight aux file the source has that we didn't write (else the dataloader
# can't process images and training hangs at step 0).
for f in os.listdir(args.model):
    src, dst = os.path.join(args.model, f), os.path.join(args.out, f)
    if (os.path.isfile(src) and not os.path.exists(dst)
            and not f.endswith(".safetensors") and f != "model.safetensors.index.json"):
        shutil.copy2(src, dst)
print("merged stage-1 checkpoint ->", args.out)
