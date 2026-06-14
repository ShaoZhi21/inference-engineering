# load the data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# create transform function that applies to all data
# converts PIL image -> tensor ([1, 28, 28] + floats) -> normalised 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# load the training and data set from MNIST
# each training data is a [tensor, label]
# the tensor is 1x28x28, 1 outermost channel array, 28 rows array, 28 values/columns in each array
train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# wrap the data sets into a data loader
train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_set,
    batch_size=1000,
    shuffle=False
)

import torch.nn as nn

# create the neural network
# flatten -> linear -> relu -> linear
# 28 x 28 = 784 input -> 128 hidden -> 10 output

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

import torch

# setup and initialise model, loss function, optimiser
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# run actual training
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch}: loss {loss.item():.4f}")

# run eval on model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"accuracy: {correct/total:.4f}")

import matplotlib.pyplot as plt

images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    preds = model(images).argmax(dim=1)

fig, axes = plt.subplots(1, 6, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(images[i].squeeze().cpu(), cmap="gray")
    ax.set_title(f"P:{preds[i].item()} T:{labels[i].item()}")
    ax.axis("off")
plt.savefig("preds.png")
print("saved preds.png")

torch.save(model.state_dict(), "mnist_net.pt")
print("saved mnist model weights")