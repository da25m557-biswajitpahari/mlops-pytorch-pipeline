import torch
from torch import nn
from torchvision.models import resnet18

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_model(architecture="simple_cnn", num_classes=10):
    if architecture == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if architecture == "resnet18":
        model = resnet18(weights=None)

        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )
        return model

    raise ValueError(
        f"Unsupported architecture: {architecture}"
    )
