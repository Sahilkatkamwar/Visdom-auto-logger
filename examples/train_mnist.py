from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from logger.logger import AutoVisdomLogger
from utils.grad_utils import get_lr


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_loaders(data_dir="./data", batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train():
    print("🚀 Training started")   # 👈 ADD THIS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader, test_loader = get_loaders()

    logger = AutoVisdomLogger(env="visdom-auto-logger")
    logger.watch(model)
    
    logger.log_metrics({"debug/test": 1.0}, step=0)   # 👈 CRITICAL
    

    global_step = 0
    epochs = 3

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            print(f"STEP: {global_step}")   # 👈 ADD THIS
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            # backward pass
            loss.backward()

            # automatic gradient logging via hooks
            print("🔥 Flushing gradients")   # 👈 ADD THIS
            logger.flush_gradients(step=global_step)

            optimizer.step()

            # training stats
            preds = logits.argmax(dim=1)
            running_loss += loss.item() * x.size(0)
            running_correct += (preds == y).sum().item()
            running_samples += x.size(0)

            # manual loss logging, but only one clean call
            if batch_idx % 20 == 0:
                print("📊 Logging metrics")   # 👈 ADD THIS
                logger.log_metrics(
                    {
                        "train/loss_step": loss.item(),
                        "train/lr": get_lr(optimizer),
                    },
                    step=global_step,
                )

            global_step += 1

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        logger.log_metrics(
            {
                "epoch/train_loss": train_loss,
                "epoch/train_acc": train_acc,
                "epoch/val_loss": val_loss,
                "epoch/val_acc": val_acc,
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    logger.close()


if __name__ == "__main__":
    train()