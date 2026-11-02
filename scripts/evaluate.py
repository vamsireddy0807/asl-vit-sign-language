"""Evaluate a trained model on the saved validation split: accuracy, weighted P/R/F1,
per-class report and a row-normalised confusion matrix.

    python scripts/evaluate.py --model-dir artifacts/asl_vit_hf_model --split artifacts/val_split.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import ViTForImageClassification, ViTImageProcessor  # noqa: E402

from asl_vit.dataset import ASLDataset, eval_transforms  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", required=True, help="CSV with image_path and label columns")
    ap.add_argument("--out", type=Path, default=Path("eval"))
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained(args.model_dir)
    model = ViTForImageClassification.from_pretrained(args.model_dir).to(device).eval()
    names = [model.config.id2label[i] for i in range(model.config.num_labels)]

    df = pd.read_csv(args.split)
    loader = DataLoader(
        ASLDataset(df, eval_transforms(processor.image_mean, processor.image_std)),
        batch_size=args.batch,
        num_workers=2,
    )
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            preds.extend(logits.argmax(-1).cpu().numpy())
            labels.extend(batch["labels"].numpy())
    preds, labels = np.array(preds), np.array(labels)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    metrics = {
        "images": int(len(labels)),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "precision_weighted": round(float(p), 4),
        "recall_weighted": round(float(r), 4),
        "f1_weighted": round(float(f1), 4),
        "per_class": classification_report(labels, preds, target_names=names, output_dict=True, zero_division=0),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(labels, preds, labels=list(range(len(names))))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(cm_norm, annot=True, fmt=".0f", cmap="Blues", xticklabels=names, yticklabels=names,
                vmin=0, vmax=100, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(args.out / "confusion_matrix.png", dpi=120)
    print({k: v for k, v in metrics.items() if k != "per_class"})


if __name__ == "__main__":
    main()
