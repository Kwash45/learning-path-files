# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os
import torch
import torch.nn as nn

CHECKPOINT_PATH = os.environ.get(
    "MNIST_CHECKPOINT",
    "/home/developer/models/mnist_model.pth",
)

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

model = MNISTModel()
ModelUnderTest = model

def load_calibration_input():
    path = "/home/developer/models/mnist_calibration/sample_0000.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing calibration sample: {path}. "
            "Download sample_0000.pt before exporting the quantized model."
        )

    x = torch.load(path, map_location="cpu")

    if isinstance(x, (tuple, list)):
        x = x[0]
    if isinstance(x, dict):
        x = next(v for v in x.values() if hasattr(v, "shape"))

    x = x.to(dtype=torch.float32)

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)

    if x.max() > 3.0:
        x = x / 255.0

    if x.min() >= 0.0 and x.max() <= 1.0:
        x = (x - 0.1307) / 0.3081

    return x.contiguous()

ModelInputs = (load_calibration_input(),)

if os.environ.get("MNIST_LOAD_CHECKPOINT", "0") == "1":
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Missing trained checkpoint: {CHECKPOINT_PATH}. "
            "Train the model before exporting."
        )

    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
