from enum import Enum
from utils.Trainer import trainer
import tomllib

with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)

def train():
    epochs = config["Train"]["epochs"]
    batch_size = config["Train"]["batch_size"]
    lr = config["Train"]["lr"]

    predicate = [
        "in",
        "on",
        "next_to",
        "on_top_of",
        "near",
        "under"
    ]

    for pred in predicate:
        neg_predicates = [p for p in predicate if p != pred]
        print(f"Training {pred} predicate")
        trainer(
            pos_predicate=pred, 
            neg_predicates=neg_predicates, 
            epoches=epochs, 
            batch_size=batch_size, 
            lr=lr
        )

if __name__ == "__main__":
    train()
