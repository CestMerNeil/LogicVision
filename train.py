"""Train multiple predicates using the Logic Tensor Network trainer.

This script reads training parameters from a configuration file and trains a set of
predicates by considering each predicate as positive while treating the others as negatives.
"""

import tomllib

from utils.Trainer import trainer

with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)


def train():
    """Train predicates based on configuration parameters.

    Reads the number of epochs, batch size, and learning rate from the configuration file,
    then iterates over a predefined list of predicates. For each predicate, it treats all the
    other predicates as negative examples and calls the trainer function.

    Returns:
        None
    """
    epochs = config["Train"]["epochs"]
    batch_size = config["Train"]["batch_size"]
    lr = config["Train"]["lr"]

    predicates = ["in", "on", "next to", "on top of", "near", "under"]

    for pred in predicates:
        neg_predicates = [p for p in predicates if p != pred]
        print(f"Training {pred} predicate")
        trainer(
            pos_predicate=pred,
            neg_predicates=neg_predicates,
            epoches=epochs,
            batch_size=batch_size,
            lr=lr,
        )


if __name__ == "__main__":
    train()
