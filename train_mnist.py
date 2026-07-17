# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause-Clear
import argparse
import gzip
import os
import struct
import urllib.request

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mnist_model import MNISTModel

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

def download(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)

def read_images(path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad image file magic: {magic}")
        data = torch.frombuffer(f.read(), dtype=torch.uint8).clone()
    data = data.view(count, 1, rows, cols).float() / 255.0
    return (data - MNIST_MEAN) / MNIST_STD

def read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad label file magic: {magic}")
        data = torch.frombuffer(f.read(), dtype=torch.uint8).clone()
    return data.long()

def load_mnist(data_dir):
    paths = {}
    for key, name in FILES.items():
        path = os.path.join(data_dir, name)
        download(BASE_URL + name, path)
        paths[key] = path

    train = TensorDataset(read_images(paths["train_images"]), read_labels(paths["train_labels"]))
    test = TensorDataset(read_images(paths["test_images"]), read_labels(paths["test_labels"]))
    return train, test

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_total += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
    return loss_total / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/developer/models/data")
    parser.add_argument("--out", default="/home/developer/models/mnist_model.pth")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)

    train_set, test_set = load_mnist(args.data_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    model = MNISTModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"epoch={epoch} test_loss={test_loss:.4f} test_acc={test_acc * 100:.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.cpu().state_dict(), args.out)
    print(f"saved {args.out}")

if __name__ == "__main__":
    main()