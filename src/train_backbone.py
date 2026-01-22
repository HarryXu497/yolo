from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from dataclasses import astuple, dataclass
import os

from utils import create_pretrain, get_device

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

DEVICE = get_device()

LEARNING_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


class TransformSubset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def pretrain(
    *,
    model: nn.Module,
    epochs: int,
    starting_epoch: int = 1,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    scheduler_path: Optional[Path | str] = None,
    optimizer_path: Optional[Path | str] = None,
    checkpoint_path: Path | str,
    checkpoints: set[int],
):
    """
    Pretrains the model backbone.

    :param model: The model to pretrain.
    :type model: nn.Module
    :param epochs: The number of epochs to train for.
    :type epochs: int
    :param starting_epoch: The epoch to start at
    :type starting_epoch: int
    :param train_loader: The data loader for the training dataset
    :type train_loader: YOLOVocDataset
    :param validation_loader: The data loader for the test dataset
    :type validation_loader: YOLOVocDataset
    :param scheduler_path: The path to the existing scheduler weights, or None to start from scratch.
    :type scheduler_path: Optional[Path | str]
    :param backbone_save_path: The path to the save the backbone weights
    :type backbone_save_path: Path | str
    :param pretrain_save_path: The path to the save the pretrain weights
    :type pretrain_save_path: Path | str
    :param scheduler_save_path: The path to the save the scheduler weights
    :type scheduler_save_path: Path | str
    :param checkpoints: The set of epochs at which to save the model weights
    :type checkpoints: set[int]
    """
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    scheduler = OneCycleLR(
        optimizer, max_lr=0.05, steps_per_epoch=len(train_loader), epochs=epochs
    )

    if optimizer_path is not None:
        optimizer.load_state_dict(torch.load(
            optimizer_path))

    if scheduler_path is not None:
        scheduler.load_state_dict(torch.load(
            scheduler_path))
    elif starting_epoch > 1:
        steps_to_skip = (starting_epoch - 1) * len(train_loader)
        for _ in range(steps_to_skip):
            scheduler.step()

    for epoch in range(starting_epoch - 1, epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader,
                    total=len(train_loader), leave=False)
        loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")

        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        val_acc = _compute_accuracy(model, validation_loader)
        print(
            f"Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")

        if (epoch + 1) in checkpoints:
            save_dir = checkpoint_path / str(epoch + 1)
            save_dir.mkdir(parents=True, exist_ok=True)

            torch.save(model.backbone.state_dict(),
                       save_dir / f"backbone.pt")
            torch.save(model.state_dict(), save_dir / f"pretrain.pt")
            torch.save(scheduler.state_dict(), save_dir / f"scheduler.pt")
            torch.save(optimizer.state_dict(), save_dir / f"optimizer.pt")


def train_val_split(train_set: Dataset, train_ratio: float) -> tuple[Subset, Subset]:
    assert 0 <= train_ratio <= 1

    total_size = len(train_set)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    return tuple(random_split(
        train_set, [train_size,
                    val_size], generator=torch.Generator().manual_seed(42)
    ))


def _compute_accuracy(model: nn.Module, loader: DataLoader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(loader,
                    total=len(loader), leave=False)
        loop.set_description(f"Validating")

        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            loop.set_postfix(score=f"{correct}/{total}",
                             acc=f"{(100 * correct / total):.2f}%")

    return 100 * correct / total


def main():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_set, val_set = train_val_split(
        torchvision.datasets.CIFAR100(
            root='./dataset/backbone', train=True, download=True),
        train_ratio=0.8
    )

    train_set = TransformSubset(train_set, transform=train_transform)
    val_set = TransformSubset(val_set, transform=val_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    model = create_pretrain(
        pretrain_path=None, num_outputs=100)
    model.to(DEVICE)

    pretrain(
        model=model,
        epochs=30,
        starting_epoch=1,
        train_loader=train_loader,
        validation_loader=val_loader,
        scheduler_path=None,
        optimizer_path=None,
        checkpoint_path="weights/checkpoints",
        checkpoints={i for i in range(0, 31, 2)}
    )


if __name__ == "__main__":
    main()
