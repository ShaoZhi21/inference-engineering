import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "data/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "data/cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 5
LEARNING_RATE = 1e-3


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_torchvision_loaders(data_dir="data", batch_size=BATCH_SIZE):
    transform = get_transform()

    train_set = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, test_loader


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    return loss.item()


def evaluate(model, data_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def save_predictions(model, test_loader, device, output_path="preds.png"):
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    model.eval()
    with torch.no_grad():
        preds = model(images).argmax(dim=1)

    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].squeeze().cpu(), cmap="gray")
        ax.set_title(f"P:{preds[i].item()} T:{labels[i].item()}")
        ax.axis("off")

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"saved {output_path}")


def run_training(
    train_loader,
    test_loader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weights_path="mnist_net.pt",
    preds_path="preds.png",
):
    device = get_device()
    model = Net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"epoch {epoch}: loss {loss:.4f}, accuracy {accuracy:.4f}")

    save_predictions(model, test_loader, device, preds_path)

    torch.save(model.state_dict(), weights_path)
    print(f"saved {weights_path}")

    return model


def main():
    train_loader, test_loader = get_torchvision_loaders()
    run_training(
        train_loader,
        test_loader,
        weights_path=Path("mnist_net.pt"),
        preds_path=Path("preds.png"),
    )


if __name__ == "__main__":
    main()
