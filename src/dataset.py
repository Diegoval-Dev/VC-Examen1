from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
):
    """
    Build train and validation DataLoaders from a directory of images.

    Supports two layouts:
    1. Pre-split: data_dir/train/<class>/ and data_dir/val/<class>/
    2. Single root: data_dir/<class>/  — will be split by val_split ratio.

    Returns:
        train_loader, val_loader, class_names, num_classes
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if train_dir.is_dir() and val_dir.is_dir():
        train_dataset = datasets.ImageFolder(
            root=str(train_dir), transform=get_train_transform()
        )
        val_dataset = datasets.ImageFolder(
            root=str(val_dir), transform=get_val_transform()
        )
    else:
        full_dataset = datasets.ImageFolder(
            root=str(data_path), transform=get_train_transform()
        )
        n_total = len(full_dataset)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_split_dataset = random_split(
            full_dataset, [n_train, n_val], generator=generator
        )
        val_split_dataset.dataset = datasets.ImageFolder(
            root=str(data_path), transform=get_val_transform()
        )
        val_dataset = val_split_dataset

    class_names = (
        train_dataset.classes
        if hasattr(train_dataset, "classes")
        else train_dataset.dataset.classes
    )
    num_classes = len(class_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, class_names, num_classes
