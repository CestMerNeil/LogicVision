import json
import tomllib

import ltn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.Relationship_Predicate import In, Near, NextTo, On, OnTopOf, Under


def auto_select_device() -> torch.device:
    """Select an available device for PyTorch operations.

    Returns:
        torch.device: The selected device (CUDA if available, then MPS, otherwise CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class Logic_Tensor_Networks:
    """Logic Tensor Networks for evaluating spatial relationships between objects.

    This class processes detector outputs to create LTN variables and manages multiple predicate networks.
    """

    def __init__(
        self,
        detector_output: dict,
        input_dim: int,
        class_labels: list,
        device: torch.device = None,
        train=True,
    ):
        """Initialize the Logic_Tensor_Networks instance.

        Args:
            detector_output (dict): Detector output containing keys like "centers", "widths", "heights", and "classes".
            input_dim (int): Dimension of the input features for the predicate networks.
            class_labels (list): List of class labels.
            device (torch.device, optional): Device to run computations on. Defaults to None.
            train (bool, optional): Flag indicating whether the network is in training mode. Defaults to True.
        """
        if device is None:
            device = auto_select_device()
        self.device = device
        print(f"Using device: {self.device}")
        self.train = train

        processed_detector_output = {}
        for key, value in detector_output.items():
            if isinstance(value, torch.Tensor):
                processed_detector_output[key] = value.to(self.device)
            elif isinstance(value, list):
                tensor_list = [
                    (
                        torch.tensor(item, dtype=torch.float, device=self.device)
                        if not isinstance(item, torch.Tensor)
                        else item.to(self.device)
                    )
                    for item in value
                ]
                processed_detector_output[key] = torch.stack(tensor_list, dim=0)
            else:
                processed_detector_output[key] = value
        self.detector_output = processed_detector_output
        self.class_labels = class_labels

        self.variables = self._variable_builder(self.detector_output)

        self.in_predicate = In(input_dim).to(self.device)
        self.on_predicate = On(input_dim).to(self.device)
        self.next_to_predicate = NextTo(input_dim).to(self.device)
        self.on_top_of_predicate = OnTopOf(input_dim).to(self.device)
        self.near_predicate = Near(input_dim).to(self.device)
        self.under_predicate = Under(input_dim).to(self.device)

        if not train:
            self.in_predicate.model.load_state_dict(
                torch.load(
                    "weights/in_predicate_weights.pth",
                    map_location=self.device,
                    weights_only=True,
                )
            )
            self.on_predicate.model.load_state_dict(
                torch.load(
                    "weights/on_predicate_weights.pth",
                    map_location=self.device,
                    weights_only=True,
                )
            )
            self.next_to_predicate.model.load_state_dict(
                torch.load(
                    "weights/next to_predicate_weights.pth",
                    map_location=self.device,
                    weights_only=True,
                )
            )
            self.on_top_of_predicate.model.load_state_dict(
                torch.load(
                    "weights/on top of_predicate_weights.pth",
                    map_location=self.device,
                    weights_only=True,
                )
            )
            self.near_predicate.model.load_state_dict(
                torch.load(
                    "weights/near_predicate_weights.pth",
                    map_location=self.device,
                    weights_only=True,
                )
            )
            self.under_predicate.model.load_state_dict(
                torch.load(
                    "weights/under_predicate_weights.pth",
                    map_location=self.device,
                    weights_only=True,
                )
            )

    def _variable_builder(self, detector_output: dict):
        """Build LTN variables from the detector output.

        Args:
            detector_output (dict): Dictionary containing keys 'centers', 'widths', 'heights', and 'classes'.

        Returns:
            dict: A dictionary mapping variable names to ltn.Variable objects.

        Raises:
            ValueError: If a required key is missing in detector_output.
        """
        variables = {}
        required_keys = ["centers", "widths", "heights", "classes"]
        for key in required_keys:
            if key not in detector_output:
                raise ValueError(f"Missing key '{key}' in detector_output.")
            value = detector_output[key]
            if isinstance(value, list):
                tensor = [
                    (
                        torch.tensor(item, dtype=torch.float, device=self.device)
                        if not isinstance(item, torch.Tensor)
                        else item.to(self.device)
                    )
                    for item in value
                ]
                concatenated = torch.stack(tensor, dim=0)
            else:
                concatenated = (
                    value
                    if isinstance(value, torch.Tensor)
                    else torch.tensor(value, dtype=torch.float, device=self.device)
                )
            if key in ["centers", "widths", "heights"] and concatenated.dim() == 1:
                concatenated = concatenated.unsqueeze(1)
            variables[key] = ltn.Variable(key, concatenated)
        if not self.train:
            print("Constructed Variables:")
            for key, var in variables.items():
                print(f"  {key}: {var.value.shape}")

        return variables

    def train_predicate(
        self,
        predicate_name: str,
        full_data: Dataset,
        epochs: int,
        batch_size: int,
        lr: float,
        val_split: float = 0.2,
    ):
        """Train a predicate network.

        Args:
            predicate_name (str): Name of the predicate to train (e.g., "in", "on", "next to").
            full_data (Dataset): Dataset containing training examples.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            lr (float): Learning rate for the optimizer.
            val_split (float, optional): Fraction of data to use for validation. Defaults to 0.2.

        Raises:
            ValueError: If an invalid predicate_name is provided.
        """
        predicate_name_lower = predicate_name.lower()
        if predicate_name_lower == "in":
            pred_net = self.in_predicate
        elif predicate_name_lower == "on":
            pred_net = self.on_predicate
        elif predicate_name_lower == "next to":
            pred_net = self.next_to_predicate
        elif predicate_name_lower == "on top of":
            pred_net = self.on_top_of_predicate
        elif predicate_name_lower == "near":
            pred_net = self.near_predicate
        elif predicate_name_lower == "under":
            pred_net = self.under_predicate
        else:
            raise ValueError(f"Invalid predicate: {predicate_name}")

        total_size = len(full_data)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_data, [train_size, val_size]
        )
        print(
            f"Total samples: {total_size}, Training samples: {train_size}, Validation samples: {val_size}"
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.AdamW(pred_net.parameters(), lr=lr)

        training_log = []

        for epoch in range(epochs):
            pred_net.train()
            epoch_train_loss = 0.0
            train_batches = 0
            for batch in train_loader:
                subj_features, obj_features, labels = batch
                subj_features = subj_features.to(self.device)
                obj_features = obj_features.to(self.device)
                labels = labels.view(-1, 1).float().to(self.device)

                subj_obj = ltn.Variable("subj", subj_features)
                obj_obj = ltn.Variable("obj", obj_features)

                outputs = pred_net(subj_obj, obj_obj)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                train_batches += 1

            avg_train_loss = (
                epoch_train_loss / train_batches if train_batches > 0 else 0.0
            )

            pred_net.eval()
            epoch_val_loss = 0.0
            val_batches = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    subj_features, obj_features, labels = batch
                    subj_features = subj_features.to(self.device)
                    obj_features = obj_features.to(self.device)
                    labels = labels.view(-1, 1).float().to(self.device)

                    subj_obj = ltn.Variable("subj", subj_features)
                    obj_obj = ltn.Variable("obj", obj_features)

                    outputs = pred_net(subj_obj, obj_obj)
                    loss = loss_fn(outputs, labels)
                    epoch_val_loss += loss.item()
                    val_batches += 1

                    preds = (outputs >= 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else 0.0
            accuracy = correct / total if total > 0 else 0.0

            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": accuracy,
            }
            training_log.append(epoch_log)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {accuracy:.4f}"
                )

        import os

        os.makedirs("weights", exist_ok=True)
        weight_path = f"weights/{predicate_name_lower}_predicate_weights.pth"
        log_path = f"weights/{predicate_name_lower}_training_log.json"
        torch.save(pred_net.model.state_dict(), weight_path)
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=4)
        print(f"Saved {predicate_name} predicate weights to '{weight_path}'.")

    def inference(
        self, subj_class: str, obj_class: str, predicate: str, threshold=0.7
    ) -> dict:
        """Perform inference using a predicate network.

        Args:
            subj_class (str): Class name of the subject.
            obj_class (str): Class name of the object.
            predicate (str): Predicate to evaluate (e.g., "in", "on", "next to").
            threshold (float, optional): Confidence threshold for determining existence. Defaults to 0.7.

        Returns:
            dict: A dictionary containing:
                - exists (bool): True if the relationship exists with confidence above the threshold.
                - confidence (float): The aggregated confidence score.
                - message (str): Status message of the inference.
                - subject_locations (dict): Locations of detected subject objects.
                - object_locations (dict): Locations of detected object objects.
                - subject_class (str): The subject class.
                - object_class (str): The object class.
                - predicate (str): The evaluated predicate.
        """
        try:
            subj_id = self.class_labels.index(subj_class)
            obj_id = self.class_labels.index(obj_class)

            subj_mask = self.detector_output["classes"] == subj_id
            obj_mask = self.detector_output["classes"] == obj_id

            if not subj_mask.any():
                return {
                    "exists": False,
                    "confidence": 0.0,
                    "message": f"{subj_class} not detected",
                }
            if not obj_mask.any():
                return {
                    "exists": False,
                    "confidence": 0.0,
                    "message": f"{obj_class} not detected",
                }

            subj_locations = {
                "centers": self.detector_output["centers"][subj_mask].cpu().tolist(),
                "widths": self.detector_output["widths"][subj_mask].cpu().tolist(),
                "heights": self.detector_output["heights"][subj_mask].cpu().tolist(),
            }
            obj_locations = {
                "centers": self.detector_output["centers"][obj_mask].cpu().tolist(),
                "widths": self.detector_output["widths"][obj_mask].cpu().tolist(),
                "heights": self.detector_output["heights"][obj_mask].cpu().tolist(),
            }

            subj_features = self._build_features(subj_mask)
            obj_features = self._build_features(obj_mask)

            subj_features = subj_features.unsqueeze(1)
            obj_features = obj_features.unsqueeze(0)

            n = subj_features.shape[0]
            m = obj_features.shape[1]
            d = subj_features.shape[2]

            subj_cart = subj_features.expand(n, m, d).reshape(-1, d)
            obj_cart = obj_features.expand(n, m, d).reshape(-1, d)

            subj_var_cart = ltn.Variable("subj_cart", subj_cart)
            obj_var_cart = ltn.Variable("obj_cart", obj_cart)

            predicate_name_lower = predicate.lower()
            if predicate_name_lower == "in":
                pred_net = self.in_predicate
            elif predicate_name_lower == "on":
                pred_net = self.on_predicate
            elif predicate_name_lower == "next to":
                pred_net = self.next_to_predicate
            elif predicate_name_lower == "on top of":
                pred_net = self.on_top_of_predicate
            elif predicate_name_lower == "near":
                pred_net = self.near_predicate
            elif predicate_name_lower == "under":
                pred_net = self.under_predicate
            else:
                raise ValueError(f"Invalid predicate: {predicate}")

            preds = pred_net(subj_var_cart, obj_var_cart)
            aggregated_score = torch.min(preds)

            return {
                "exists": aggregated_score.item() >= threshold,
                "confidence": round(aggregated_score.item(), 3),
                "message": "Inference successful",
                "subject_locations": subj_locations,
                "object_locations": obj_locations,
                "subject_class": subj_class,
                "object_class": obj_class,
                "predicate": predicate,
            }

        except Exception as e:
            return {"exists": False, "confidence": 0.0, "message": str(e)}

    def _build_features(self, mask: torch.Tensor) -> torch.Tensor:
        """Build features from the detector output based on a mask.

        Args:
            mask (torch.Tensor): Boolean mask for selecting objects.

        Returns:
            torch.Tensor: Concatenated tensor of features (centers, widths, heights, and classes).
        """
        features = torch.cat(
            [
                self.detector_output["centers"][mask],
                self.detector_output["widths"][mask].unsqueeze(1),
                self.detector_output["heights"][mask].unsqueeze(1),
                self.detector_output["classes"][mask].unsqueeze(1).float(),
            ],
            dim=1,
        )
        return features.to(self.device)


if __name__ == "__main__":
    import json

    pos_predicate = "ON"
    neg_predicates = [
        "wears",
        "has",
        "next to",
        "on top of",
        "in",
        "behind",
        "holding",
        "parked on",
        "by",
    ]

    from utils.DataLoader import RelationshipDataset

    train_dataset = RelationshipDataset(
        relationships_json_path="data/relationships.json",
        image_meta_json_path="data/image_data.json",
        pos_predicate=pos_predicate,
        neg_predicates=neg_predicates,
    )

    num_obj = 10
    device = auto_select_device()
    detector_output = {
        "centers": torch.randn(num_obj, 2, device=device),
        "widths": torch.randn(num_obj, device=device),
        "heights": torch.randn(num_obj, device=device),
        "classes": torch.randint(0, 100, (num_obj,), device=device),
    }
    class_labels = list(range(100))
    input_dim = 5

    ltn_network = Logic_Tensor_Networks(
        detector_output, input_dim, class_labels, device=device
    )
    ltn_network.train_dataset = train_dataset

    ltn_network.train_predicate(
        predicate_name="on",
        train_data=train_dataset,
        epochs=100,
        batch_size=1024,
        lr=0.001,
    )
