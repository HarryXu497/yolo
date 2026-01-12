import torch
import torch.nn as nn

from yolo.backbone import YOLOBackbone


class YOLOPretrain(nn.Module):
    def __init__(self, backbone: YOLOBackbone):
        super().__init__()
        self._backbone = backbone
        self._avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._classifier = nn.Linear(in_features=1024, out_features=1000)

    def forward(self, x):
        x = self._backbone(x)
        x = self._avg_pool(x)
        x = torch.flatten(x, dim=1)
        return self._classifier(x)

    def pretrain(self, epochs: int, train_loader, validation_loader, device):
        self.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.parameters(), lr=10e-2, momentum=0.9, weight_decay=5e-4)

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                logits = self.forward(images)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            val_acc = self._compute_accuracy(validation_loader, device)
            print(
                f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")

    @property
    def backbone(self):
        return self._backbone
    
    def _compute_accuracy(self, loader, device):
        self.eval()
        
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits = self.forward(images)
                preds = torch.argmax(logits, dim=1)
                correct += torch.sum(preds == labels)
                total += labels.size(0)
                
        return 100 * correct / total
