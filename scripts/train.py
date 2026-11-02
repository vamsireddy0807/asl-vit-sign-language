"""Two-phase ViT fine-tuning for 29-class ASL alphabet recognition.

Phase 1 trains the classifier head with the backbone frozen; phase 2 also unfreezes the
last two transformer blocks at a low learning rate.

    kaggle datasets download -d debashishsau/aslamerican-sign-language-aplhabet-dataset -p data --unzip
    python scripts/train.py --train-dir data/ASL_Alphabet_Dataset/asl_alphabet_train --out artifacts/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import (  # noqa: E402
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

from asl_vit import MODEL_NAME  # noqa: E402
from asl_vit.data import build_index, label_maps, sample_and_split, save_label_mapping  # noqa: E402
from asl_vit.dataset import ASLDataset, eval_transforms, train_transforms  # noqa: E402


def args_for(out: Path, phase: int, epochs: int, lr: float, batch: int, accum: int, warmup: int, wd: float):
    return TrainingArguments(
        output_dir=str(out / f"phase{phase}"),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=accum,  # effective batch 64 in both phases
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup,
        weight_decay=wd,
        label_smoothing_factor=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-ft", type=int, default=10)
    ap.add_argument("--unfreeze-blocks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    df = build_index(args.train_dir)
    label_to_idx, idx_to_label = label_maps(df)
    train_df, val_df = sample_and_split(df, label_to_idx, per_class=args.per_class, seed=args.seed)
    train_df.to_csv(args.out / "train_split.csv", index=False)
    val_df.to_csv(args.out / "val_split.csv", index=False)
    save_label_mapping(args.out / "label_mapping.json", label_to_idx, MODEL_NAME)
    print(f"{len(df):,} images, {len(label_to_idx)} classes -> train {len(train_df):,} / val {len(val_df):,}")

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    mean, std = processor.image_mean, processor.image_std
    train_ds = ASLDataset(train_df, train_transforms(mean, std))
    val_ds = ASLDataset(val_df, eval_transforms(mean, std))

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_to_idx), ignore_mismatched_sizes=True
    )
    model.config.id2label = {i: label for i, label in idx_to_label.items()}
    model.config.label2id = dict(label_to_idx)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return accuracy.compute(predictions=np.argmax(logits, axis=-1), references=labels)

    # Phase 1: classifier head only
    for p in model.vit.parameters():
        p.requires_grad = False
    trainer = Trainer(
        model=model,
        args=args_for(args.out, 1, args.epochs_head, 5e-4, batch=32, accum=2, warmup=100, wd=0.05),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    phase1 = trainer.evaluate()

    # Phase 2: unfreeze the last N transformer blocks, very low LR to avoid forgetting
    for block in model.vit.encoder.layer[-args.unfreeze_blocks :]:
        for p in block.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"phase 2 trainable parameters: {trainable:,}")
    trainer = Trainer(
        model=model,
        args=args_for(args.out, 2, args.epochs_ft, 2e-5, batch=16, accum=4, warmup=200, wd=0.01),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    phase2 = trainer.evaluate()

    model_dir = args.out / "asl_vit_hf_model"
    trainer.model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    summary = {"phase1": phase1, "phase2": phase2, "model_dir": str(model_dir)}
    (args.out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
