import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import create_backbone, get_device
from yolo.pretrain import YOLOPretrain

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

DEVICE = get_device()


def test(model: nn.Module, loader: DataLoader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(loader,
                    total=len(loader), leave=False)
        loop.set_description(f"Testing")

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

    test_set = torchvision.datasets.CIFAR100(
        root='./dataset/backbone', train=False, download=True, transform=transform)
    
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    backbone = create_backbone("weights/backbone/backbone-20.pt")

    model = YOLOPretrain(backbone=backbone, num_outputs=100)
    model.load_state_dict(torch.load("weights/pretrain/pretrain-20.pt"))
    model.to(DEVICE)

    result = test(model=model, loader=test_loader)
    print(result)


if __name__ == "__main__":
    main()
