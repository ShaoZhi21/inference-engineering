from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from train import BATCH_SIZE, TEST_BATCH_SIZE, get_transform, run_training


class HFMNISTDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.transform = get_transform()

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        row = self.split[idx]
        image = self.transform(row["image"])
        label = row["label"]
        return image, label


def get_hf_loaders(cache_dir="data/hf", batch_size=BATCH_SIZE):
    hf_ds = load_dataset("ylecun/mnist", cache_dir=cache_dir)

    train_set = HFMNISTDataset(hf_ds["train"])
    test_set = HFMNISTDataset(hf_ds["test"])

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


def main():
    train_loader, test_loader = get_hf_loaders()

    images, labels = next(iter(train_loader))
    print(f"batch images shape: {images.shape}")
    print(f"batch labels shape: {labels.shape}")

    run_training(
        train_loader,
        test_loader,
        weights_path="hf_mnist_net.pt",
        preds_path="hf_preds.png",
    )


if __name__ == "__main__":
    main()
