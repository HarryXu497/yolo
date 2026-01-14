from pathlib import Path
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from utils import create_backbone, get_device
from yolo.pretrain import YOLOPretrain

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

DEVICE = get_device()

LEARNING_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


def pretrain(
    *,
    model: nn.Module,
    starting_epoch: int = 1,
    epochs: int,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    backbone_save_path: Path | str,
    pretrain_save_path: Path | str,
    checkpoints: set[int],
):
    if isinstance(backbone_save_path, str):
        backbone_save_path = Path(backbone_save_path)

    if isinstance(pretrain_save_path, str):
        pretrain_save_path = Path(pretrain_save_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    scheduler = OneCycleLR(
        optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=epochs
    )

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
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        val_acc = _compute_accuracy(model, validation_loader)
        print(
            f"Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")
        
        if (epoch + 1) in checkpoints:
            torch.save(model.backbone.state_dict(), Path.joinpath(backbone_save_path, f"backbone-{epoch + 1}.pt"))
            torch.save(model.state_dict(), Path.joinpath(pretrain_save_path, f"pretrain-{epoch + 1}.pt"))


def train_val_split(train_set: Dataset, train_ratio: float) -> tuple[Subset, Subset]:
    assert 0 <= train_ratio <= 1

    total_size = len(train_set)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size


    return tuple(random_split(
        train_set, [train_size, val_size], generator=torch.Generator().manual_seed(42)
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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_set, val_set = train_val_split(
        torchvision.datasets.CIFAR100(
        root='./dataset/backbone', train=True, download=True, transform=transform),
        train_ratio=0.8
    )

    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_set,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    backbone = create_backbone()

    model = YOLOPretrain(backbone=backbone, num_outputs=100)
    model.to(DEVICE)

    pretrain(
        model=model,
        epochs=20,
        starting_epoch=1,
        train_loader=train_loader,
        validation_loader=val_loader,
        backbone_save_path="weights/backbone",
        pretrain_save_path="weights/pretrain",
        checkpoints={i for i in range(0, 21, 2)}
    )


if __name__ == "__main__":
    main()
