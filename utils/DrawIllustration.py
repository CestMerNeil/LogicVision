import json

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_metrics(file_path, save_path=None, label=None):
    with open(file_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame([entry for entry in data if "epoch" in entry])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="orange")
    axes[0].plot(df["epoch"], df["val_loss"], label="Validation Loss", color="brown")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        df["epoch"], df["val_accuracy"], label="Validation Accuracy", color="green"
    )
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].plot(df["epoch"], df["learning_rate"], label="Learning Rate", color="red")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].legend()

    plt.suptitle("Training Metrics - {}".format(label))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    plot_training_metrics(
        "logs/in_training_log_20250302_011109.json",
        "README/images/In Training Metrics.png",
        "In Predicate",
    )
    plot_training_metrics(
        "logs/near_training_log_20250302_050857.json",
        "README/images/Near Training Metrics.png",
        "Near Predicate",
    )
    plot_training_metrics(
        "logs/next to_training_log_20250302_030329.json",
        "README/images/Next To Training Metrics.png",
        "Next To Predicate",
    )
    plot_training_metrics(
        "logs/on top of_training_log_20250302_031202.json",
        "README/images/On Top Of Training Metrics.png",
        "On Top Of Predicate",
    )
    plot_training_metrics(
        "logs/on_training_log_20250302_014520.json",
        "README/images/On Training Metrics.png",
        "On Predicate",
    )
    plot_training_metrics(
        "logs/under_training_log_20250302_033035.json",
        "README/images/Under Training Metrics.png",
        "Under Predicate",
    )
