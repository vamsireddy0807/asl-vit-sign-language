"""PyTorch dataset and the augmentation pipelines used in training."""

from __future__ import annotations

import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMG_SIZE = 224


def train_transforms(mean, std, img_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size + 20, img_size + 20)),  # slight over-size for random crop
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(p=0.3),  # mild: not every ASL sign is symmetric
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def eval_transforms(mean, std, img_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])


class ASLDataset(Dataset):
    """Yields ``{"pixel_values", "labels"}`` dicts, as the Hugging Face Trainer expects."""

    def __init__(self, dataframe: pd.DataFrame, transform) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return {"pixel_values": self.transform(image), "labels": int(row["label"])}
