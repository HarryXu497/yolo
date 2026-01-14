from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import create_model, get_device
from yolo.dataset import YOLOVocDataset
from yolo.loss import YOLOLoss

LEARNING_RATE = 2e-5
DEVICE = get_device()
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0005
EPOCHS = 100


def train(
    *,
    model: nn.Module,
    epochs: int,
    starting_epoch: int = 1,
    train_loader: YOLOVocDataset,
    test_loader: YOLOVocDataset,
    loss_fn: nn.Module,
    save_path: Path | str,
    checkpoints: set[int]
):
    """
    Trains the model.

    :param model: The model to train.
    :type model: nn.Module
    :param epochs: The number of epochs to train for.
    :type epochs: int
    :param train_loader: The data loader for the training dataset
    :type train_loader: YOLOVocDataset
    :param test_loader: The data loader for the test dataset
    :type test_loader: YOLOVocDataset
    :param loss_fn: The loss function to use
    :type loss_fn: nn.Module
    :param save_path: The path to save the model weights
    :type save_path: Path | str
    :param checkpoints: The set of epochs at which to save the model weights
    :type checkpoints: set[int]
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs
    )

    for epoch in range(starting_epoch - 1, epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, total=len(train_loader), leave=False)
        loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")

        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_acc = _compute_accuracy(model, test_loader)
        print(
            f"Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")

        if (epoch + 1) in checkpoints:
            torch.save(model.state_dict(), Path.joinpath(
                save_path, f"train-{epoch + 1}.pt"))


def _compute_accuracy(model: nn.Module, test_loader: DataLoader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(test_loader,
                    total=len(test_loader), leave=False)
        loop.set_description(f"Evaluating")

        for images, targets in loop:
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            out = model(images)

            exists_box = targets[..., 4]

            predictions = out[..., 10:]
            actual = targets[..., 10:]

            pred_classes = torch.argmax(predictions, dim=-1)
            actual_classes = torch.argmax(actual, dim=-1)

            # Only consider loss from cells where an object exists
            correct_mask = (pred_classes == actual_classes) * exists_box

            correct += torch.sum(correct_mask).item()
            total += torch.sum(exists_box).item()

            if total > 0:
                loop.set_postfix(acc=f"{(100 * correct / total):.2f}%")

    return 100 * correct / total


def main():
    train_dataset = YOLOVocDataset(
        root="./dataset/yolo/train_val", year="2012", image_set="train")
    test_dataset = YOLOVocDataset(
        root="./dataset/yolo/train_val", year="2012", image_set="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    model = create_model(
        backbone_path="weights/backbone/backbone-20.pt",
        model_path="weights/train/train-1.pt"
    )
    model.to(DEVICE)

    loss_fn = YOLOLoss()
    loss_fn.to(DEVICE)

    train(
        model=model,
        epochs=20,
        starting_epoch=3,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        save_path="weights/train",
        checkpoints={i for i in range(0, 41, 2)}
    )


if __name__ == "__main__":
    main()
