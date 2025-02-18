import os
import json
import torch
import tomllib

from pathlib import Path
from torch.utils.data import DataLoader
from models.Logic_Tensor_Networks import Logic_Tensor_Networks
from utils.DataLoader import RelationshipDataset

with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)

def trainer(pos_predicate: str, neg_predicates: list, epoches=1, batch_size=512, lr=0.001):
    # Initialize the dataset
    relationships_path = config["Trainer"]["relationships_path"]
    image_meta_path = config["Trainer"]["image_meta_path"]

    train_dataset = RelationshipDataset(
        relationships_json_path=relationships_path,
        image_meta_json_path=image_meta_path,
        pos_predicate=pos_predicate,
        neg_predicates=neg_predicates
    )

    # Print the dataset examples
    print("Length of train_dataset: ", end="")
    print(len(train_dataset))
    print("train_dataset[0]: ", end="")
    print(train_dataset[0])
    print("train_dataset[1]: ", end="")
    print(train_dataset[1])
    print("train_dataset[2]: ", end="")
    print(train_dataset[2])

    # Initialize the detector output
    num_obj = 100
    class_labels = list(range(100))
    input_dim = 5

    detector_output = {
        "centers": torch.randn(num_obj, 2),
        "widths": torch.randn(num_obj, 1),
        "heights": torch.randn(num_obj, 1),
        "classes": torch.randint(0, 10, (num_obj,))
    }

    ltn_network = Logic_Tensor_Networks(detector_output, input_dim, class_labels)
    ltn_network.train_predicate(
        predicate_name=pos_predicate, 
        full_data=train_dataset, 
        epochs=epoches, 
        batch_size=batch_size, 
        lr=lr
    )

if __name__ == "__main__":
    pos_predicate = "near"
    neg_predicates = ["is"]
    trainer(pos_predicate, neg_predicates)

