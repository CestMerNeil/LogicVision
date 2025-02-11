from enum import Enum
from utils.Trainer import trainer
import tomllib

with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)

class predicate(Enum):
    In = "in",
    On = "on",
    NextTo = "next_to",
    OnTopOf = "on_top_of",
    Near = "near",
    Under = "under"

def train():
    epoches = config["Train"]["epochs"]
    batch_size = config["Train"]["batch_size"]
    lr = config["Train"]["lr"]

    for pred in predicate:
        neg_predicates = [p for p in predicate if p != pred]
        print(f"Training {pred} predicate")
        trainer(pred, neg_predicates, epoches, batch_size, lr)

if __name__ == "__main__":
    train()
