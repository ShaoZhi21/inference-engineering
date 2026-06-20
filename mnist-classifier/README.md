---
library_name: pytorch
tags:
- mnist
- image-classification
- pytorch
datasets:
- ylecun/mnist
metrics:
- accuracy
---

# MNIST Classifier

This is a small PyTorch classifier trained on the MNIST handwritten digit dataset.
It was built as a Week 0 PyTorch fundamentals exercise: dataset loading, `DataLoader`,
`nn.Module`, training loop, evaluation, and saving model weights.

## Model Details

The model is a simple feed-forward neural network:

- Input: 28x28 grayscale MNIST image
- Flatten: 784 features
- Linear: 784 -> 128
- Activation: ReLU
- Linear: 128 -> 10
- Output: digit class logits for labels 0-9

The uploaded `hf_mnist_net.pt` file is a PyTorch `state_dict`, not a full Hugging
Face Transformers model package. To use it, recreate the same model architecture
and then load the state dict.

## Dataset

Trained on [`ylecun/mnist`](https://huggingface.co/datasets/ylecun/mnist).

Images are converted to tensors and normalized with:

```python
transforms.Normalize((0.1307,), (0.3081,))
```

## Results

Local test accuracy after 5 epochs was approximately 97%.

## Usage

```python
import torch
import torch.nn as nn


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


model = Net()
state_dict = torch.load("hf_mnist_net.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
```

## Limitations

This model is for learning and demonstration only. It was trained on MNIST and is
not intended for production use or for recognizing arbitrary real-world handwriting.
