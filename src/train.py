import json
import os
from pathlib import Path

import torch
from torch import nn
import yaml

from dataset import get_dataloaders
from model import get_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    config_path = os.getenv(
        "TRAINING_CONFIG", "configs/training_config.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 0),
    )

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / config["output"]["model_name"]

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    print(json.dumps({
        "event": "training_started",
        "device": str(device),
        "epochs": config["training"]["epochs"],
    }), flush=True)

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )

        print(json.dumps({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_accuracy, 4),
        }), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
                "class_names": train_loader.dataset.classes,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }, checkpoint_path)

            print(json.dumps({
                "event": "checkpoint_saved",
                "path": str(checkpoint_path),
            }), flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({
                    "event": "early_stopping",
                    "epoch": epoch,
                }), flush=True)
                break

    print(json.dumps({
        "event": "training_complete",
        "best_val_loss": best_val_loss,
    }), flush=True)


if __name__ == "__main__":
    main()
