"""Script to train a Logic Tensor Network predicate using a relationship dataset.

This script loads configuration settings from a TOML file, initializes a relationship
dataset, and trains a predicate using a Logic Tensor Network.
"""

import json
import os
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.Logic_Tensor_Networks import Logic_Tensor_Networks
from utils.DataLoader import RelationshipDataset

with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)


def trainer(
    pos_predicate: str,
    neg_predicates: list,
    epoches: int = 1,
    batch_size: int = 512,
    lr: float = 0.001,
):
    """Train a predicate using a relationship dataset and a Logic Tensor Network.

    This function initializes a RelationshipDataset using the paths specified in the
    configuration file, prints sample entries from the dataset, creates a dummy detector
    output, and trains the predicate network using the provided training parameters.

    Args:
        pos_predicate (str): The positive predicate to train (e.g., "near").
        neg_predicates (list): A list of negative predicates.
        epoches (int, optional): Number of training epochs. Defaults to 1.
        batch_size (int, optional): Batch size for training. Defaults to 512.
        lr (float, optional): Learning rate for training. Defaults to 0.001.

    Returns:
        None
    """
    relationships_path = config["Trainer"]["relationships_path"]
    image_meta_path = config["Trainer"]["image_meta_path"]

    lr_factor = config["Trainer"]["lr_factor"]
    lr_patience = config["Trainer"]["lr_patience"]
    lr_min = config["Trainer"]["lr_min"]
    patience = config["Trainer"]["patience"]

    train_dataset = RelationshipDataset(
        relationships_json_path=relationships_path,
        image_meta_json_path=image_meta_path,
        pos_predicate=pos_predicate,
        # neg_predicates=neg_predicates,
    )

    print("Length of train_dataset: ", end="")
    print(len(train_dataset))
    print("train_dataset[0]: ", end="")
    print(train_dataset[0])
    print("train_dataset[1]: ", end="")
    print(train_dataset[1])
    print("train_dataset[2]: ", end="")
    print(train_dataset[2])

    num_obj = 100
    class_labels = list(range(100))
    input_dim = 5

    detector_output = {
        "centers": torch.randn(num_obj, 2),
        "widths": torch.randn(num_obj, 1),
        "heights": torch.randn(num_obj, 1),
        "classes": torch.randint(0, 10, (num_obj,)),
    }

    ltn_network = Logic_Tensor_Networks(detector_output, input_dim, class_labels)
    ltn_network.train_predicate(
        predicate_name=pos_predicate,
        full_data=train_dataset,
        epochs=epoches,
        batch_size=batch_size,
        lr=lr,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        lr_min=lr_min,
        patience=patience,
    )


if __name__ == "__main__":
    pos_predicate = "near"
    neg_predicates = ["is"]
    trainer(pos_predicate, neg_predicates)
