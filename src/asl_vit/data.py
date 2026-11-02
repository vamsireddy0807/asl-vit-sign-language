"""Dataset indexing, deterministic label mapping and stratified sampling (pandas only)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def build_index(train_dir: str) -> pd.DataFrame:
    """One row per image: ``image_path``, ``label_name`` (the class folder name)."""
    rows = []
    for label in sorted(os.listdir(train_dir)):
        folder = os.path.join(train_dir, label)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(IMAGE_EXTS):
                rows.append({"image_path": os.path.join(folder, name), "label_name": label})
    df = pd.DataFrame(rows, columns=["image_path", "label_name"])
    if df.empty:
        raise FileNotFoundError(f"no images found under {train_dir}")
    return df


def label_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Sorted, fixed label order so indices never depend on filesystem order."""
    order = sorted(df["label_name"].unique())
    to_idx = {label: i for i, label in enumerate(order)}
    return to_idx, {i: label for label, i in to_idx.items()}


def sample_and_split(
    df: pd.DataFrame,
    label_to_idx: Dict[str, int],
    per_class: int = 1500,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample ``per_class`` images from every class, encode labels, stratified split."""
    counts = df["label_name"].value_counts()
    short = counts[counts < per_class]
    if not short.empty:
        raise ValueError(f"classes with fewer than {per_class} images: {short.to_dict()}")
    sampled = (
        df.groupby("label_name", group_keys=False)[["image_path", "label_name"]]
        .apply(lambda g: g.sample(n=per_class, random_state=seed))
        .reset_index(drop=True)
    )
    sampled["label"] = sampled["label_name"].map(label_to_idx)
    train_df, val_df = train_test_split(
        sampled, test_size=val_fraction, stratify=sampled["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def save_label_mapping(path: Path, label_to_idx: Dict[str, int], model_name: str) -> None:
    idx_to_label = {str(i): label for label, i in label_to_idx.items()}
    path.write_text(
        json.dumps(
            {"label_to_idx": label_to_idx, "idx_to_label": idx_to_label, "num_classes": len(label_to_idx),
             "model_name": model_name},
            indent=2,
        )
    )
