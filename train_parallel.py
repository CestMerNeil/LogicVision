import json
import os
import tomllib

import ltn
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from models.Logic_Tensor_Networks import Logic_Tensor_Networks
from utils.DataLoader import RelationshipDataset

# Load configuration
with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)


def train_combined():
    """Train all predicates together to maximize GPU utilization."""
    # Read configuration
    epochs = config["Train"]["epochs"]
    batch_size = config["Train"]["batch_size"]
    lr = config["Train"]["lr"]

    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List of predicates to train
    predicates = ["in", "on", "next to", "on top of", "near", "under"]

    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Create dummy detector output for LTN initialization
    num_obj = 10
    detector_output = {
        "centers": torch.randn(num_obj, 2, device=device),
        "widths": torch.randn(num_obj, device=device),
        "heights": torch.randn(num_obj, device=device),
        "classes": torch.randint(0, 10, (num_obj,), device=device),
    }

    # Create LTN network
    class_labels = list(range(100))
    input_dim = 5
    ltn_network = Logic_Tensor_Networks(
        detector_output, input_dim, class_labels, device=device
    )

    # Dictionary to store predicate networks
    pred_networks = {
        "in": ltn_network.in_predicate,
        "on": ltn_network.on_predicate,
        "next to": ltn_network.next_to_predicate,
        "on top of": ltn_network.on_top_of_predicate,
        "near": ltn_network.near_predicate,
        "under": ltn_network.under_predicate,
    }

    # Create loss functions and optimizers for each predicate
    loss_fn = nn.BCELoss()
    optimizers = {
        pred: torch.optim.AdamW(pred_networks[pred].parameters(), lr=lr)
        for pred in predicates
    }

    # Load all datasets simultaneously
    relationships_path = config["Trainer"]["relationships_path"]
    image_meta_path = config["Trainer"]["image_meta_path"]

    # Create dataset for each predicate
    datasets = {}
    for pred in predicates:
        neg_predicates = [p for p in predicates if p != pred]
        datasets[pred] = RelationshipDataset(
            relationships_json_path=relationships_path,
            image_meta_json_path=image_meta_path,
            pos_predicate=pred,
            # neg_predicates=neg_predicates,
        )

    # Create training and validation data loaders with optimizations
    train_loaders = {}
    val_loaders = {}

    for pred in predicates:
        train_size = int(0.8 * len(datasets[pred]))
        val_size = len(datasets[pred]) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            datasets[pred], [train_size, val_size]
        )

        train_loaders[pred] = DataLoader(
            train_subset,
            batch_size=batch_size // len(predicates),
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        val_loaders[pred] = DataLoader(
            val_subset,
            batch_size=batch_size // len(predicates),
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

    # Training loop
    print(f"Training all predicates together on {device}")
    training_logs = {pred: [] for pred in predicates}

    for epoch in range(epochs):
        for pred in predicates:
            pred_networks[pred].train()

        epoch_train_losses = {pred: 0.0 for pred in predicates}
        train_batches = {pred: 0 for pred in predicates}
        data_iters = {pred: iter(train_loaders[pred]) for pred in predicates}
        active_predicates = set(predicates)

        while active_predicates:
            for pred in list(active_predicates):
                try:
                    batch = next(data_iters[pred])
                    subj_features, obj_features, labels = batch
                    subj_features, obj_features, labels = (
                        subj_features.to(device, non_blocking=True),
                        obj_features.to(device, non_blocking=True),
                        labels.view(-1, 1).float().to(device, non_blocking=True),
                    )

                    subj_obj = ltn.Variable("subj", subj_features)
                    obj_obj = ltn.Variable("obj", obj_features)

                    optimizers[pred].zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast():
                        outputs = pred_networks[pred](subj_obj, obj_obj)
                        loss = loss_fn(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizers[pred])
                    scaler.update()

                    epoch_train_losses[pred] += loss.item()
                    train_batches[pred] += 1
                except StopIteration:
                    active_predicates.remove(pred)

        for pred in predicates:
            pred_networks[pred].eval()
            epoch_val_loss, val_batches, correct, total = 0.0, 0, 0, 0

            with torch.no_grad():
                for batch in val_loaders[pred]:
                    subj_features, obj_features, labels = batch
                    subj_features, obj_features, labels = (
                        subj_features.to(device, non_blocking=True),
                        obj_features.to(device, non_blocking=True),
                        labels.view(-1, 1).float().to(device, non_blocking=True),
                    )

                    outputs = pred_networks[pred](subj_obj, obj_obj)
                    loss = loss_fn(outputs, labels)
                    epoch_val_loss += loss.item()
                    val_batches += 1

                    preds = (outputs >= 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            training_logs[pred].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_train_losses[pred]
                    / max(train_batches[pred], 1),
                    "val_loss": epoch_val_loss / max(val_batches, 1),
                    "val_accuracy": correct / max(total, 1),
                }
            )

    os.makedirs("weights", exist_ok=True)
    for pred in predicates:
        torch.save(
            pred_networks[pred].model.state_dict(),
            f"weights/{pred.replace(' ', '_').lower()}_weights.pth",
        )
        with open(f"weights/{pred.replace(' ', '_').lower()}_log.json", "w") as f:
            json.dump(training_logs[pred], f, indent=4)


if __name__ == "__main__":
    train_combined()
