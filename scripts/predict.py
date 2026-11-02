"""Predict the ASL letter (A-Z, space, del, nothing) for one or more images, with top-k scores.

    python scripts/predict.py --model-dir artifacts/asl_vit_hf_model --image hand.jpg --topk 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import ViTForImageClassification, ViTImageProcessor  # noqa: E402

from asl_vit.dataset import eval_transforms  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", required=True, nargs="+")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained(args.model_dir)
    model = ViTForImageClassification.from_pretrained(args.model_dir).to(device).eval()
    transform = eval_transforms(processor.image_mean, processor.image_std)

    for path in args.image:
        x = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(pixel_values=x).logits, dim=-1).squeeze(0).cpu()
        top = torch.topk(probs, k=min(args.topk, probs.numel()))
        ranked = [
            (model.config.id2label[int(i)], round(float(p) * 100, 2))
            for p, i in zip(top.values, top.indices, strict=True)
        ]
        print(json.dumps({"image": path, "prediction": ranked[0][0], "confidence": ranked[0][1], "top_k": ranked}))


if __name__ == "__main__":
    main()
