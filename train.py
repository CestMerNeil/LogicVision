import json
import torch
import tomllib
import os
from pathlib import Path

from torch.utils.data import DataLoader
from models.Logic_Tensor_Networks import Logic_Tensor_Networks
from utils.DataLoader import RelationshipDataset

PROJECT_ROOT = Path(__file__).parent

def main():
    # Load Config
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    pos_predicate = "near"
    neg_predicate = "is"

    # relationships_path = os.path.join(PROJECT_ROOT, "data", "relationships.json")
    # image_meta_path = os.path.join(PROJECT_ROOT, "data", "image_data.json")
    relationships_path = "/root/autodl-tmp/relationships_full.json"
    image_meta_path = "/root/autodl-tmp/image_data_full.json"

    train_dataset = RelationshipDataset(
        relationships_json_path=relationships_path,
        image_meta_json_path=image_meta_path, 
        pos_predicate=pos_predicate,
        neg_predicates=neg_predicate
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

    detector_output = {
        "centers": torch.randn(num_obj, 2),
        "widths": torch.randn(num_obj, 1),
        "heights": torch.randn(num_obj, 1),
        "classes": torch.randint(0, 10, (num_obj,))
    }

    class_labels = list(range(100))

    input_dim = 5 # center2d, width, height, class
    ltn_network = Logic_Tensor_Networks(detector_output, input_dim, class_labels)

    epochs = 100
    batch_size = 1024
    lr = 0.001
    ltn_network.train_predicate(
        pos_predicate, 
        full_data=train_dataset,
        epochs=epochs, 
        batch_size=batch_size,
        lr=lr
        )

if __name__ == "__main__":
    main()

    