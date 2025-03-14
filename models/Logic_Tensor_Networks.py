import json
import logging
import tomllib

import ltn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.Relationship_Predicate import In, Near, NextTo, On, OnTopOf, Under

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("LogicVision")


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
        early_stopping: bool = True,
        patience: int = 50,
        monitor: str = "val_loss",
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
        lr_scheduler: bool = True,
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        lr_min: float = 1e-6,
        log_dir: str = "logs",
    ):
        """Train a predicate network.

        Args:
            predicate_name (str): Name of the predicate to train (e.g., "in", "on", "next to").
            full_data (Dataset): Dataset containing training examples.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            lr (float): Learning rate for the optimizer.
            val_split (float, optional): Fraction of data to use for validation. Defaults to 0.2.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
            patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 10.
            monitor (str, optional): Metric to monitor ('val_loss' or 'val_accuracy'). Defaults to "val_loss".
            min_delta (float, optional): Minimum change to qualify as improvement. Defaults to 0.001.
            restore_best_weights (bool, optional): Whether to restore model to best weights. Defaults to True.
            lr_scheduler (bool, optional): Whether to use learning rate scheduler. Defaults to True.
            lr_factor (float, optional): Factor by which to reduce learning rate. Defaults to 0.5.
            lr_patience (int, optional): Epochs to wait before reducing learning rate. Defaults to 5.
            lr_min (float, optional): Minimum learning rate. Defaults to 1e-6.
            log_dir (str, optional): Directory to store log files. Defaults to "logs".

        Raises:
            ValueError: If an invalid predicate_name or monitor is provided.
        """
        import os
        from datetime import datetime

        predicate_name_lower = predicate_name.lower()

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            log_dir, f"{predicate_name_lower}_training_{timestamp}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        logger.info(f"Starting training for predicate: {predicate_name}")

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
            logger.error(f"Invalid predicate: {predicate_name}")
            raise ValueError(f"Invalid predicate: {predicate_name}")

        if monitor not in ["val_loss", "val_accuracy"]:
            logger.error(
                f"Invalid monitor: {monitor}. Must be 'val_loss' or 'val_accuracy'"
            )
            raise ValueError(
                f"Invalid monitor: {monitor}. Must be 'val_loss' or 'val_accuracy'"
            )

        total_size = len(full_data)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_data, [train_size, val_size]
        )
        logger.info(
            f"Total samples: {total_size}, Training samples: {train_size}, Validation samples: {val_size}"
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.AdamW(pred_net.parameters(), lr=lr)

        scheduler = None
        if lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min" if monitor == "val_loss" else "max",
                factor=lr_factor,
                patience=lr_patience,
                verbose=False,
                min_lr=lr_min,
            )
            logger.info(
                f"Using learning rate scheduler: factor={lr_factor}, patience={lr_patience}, min_lr={lr_min}"
            )

        training_log = []

        best_metric = float("inf") if monitor == "val_loss" else -float("inf")
        best_epoch = 0
        wait = 0
        best_state_dict = None

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

            if scheduler is not None:
                metric_for_scheduler = (
                    avg_val_loss if monitor == "val_loss" else -accuracy
                )
                scheduler.step(metric_for_scheduler)
                current_lr = optimizer.param_groups[0]["lr"]
                if current_lr != lr:
                    logger.info(f"Learning rate adjusted to {current_lr:.6f}")
            else:
                current_lr = lr

            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": accuracy,
                "learning_rate": current_lr,
            }
            training_log.append(epoch_log)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, "
                    f"Val Accuracy = {accuracy:.4f}, LR = {current_lr:.6f}"
                )

            if early_stopping:
                current_metric = avg_val_loss if monitor == "val_loss" else accuracy
                improved = False

                if monitor == "val_loss":
                    improved = current_metric < best_metric - min_delta
                else:  # val_accuracy
                    improved = current_metric > best_metric + min_delta

                if improved:
                    best_metric = current_metric
                    best_epoch = epoch
                    wait = 0
                    logger.info(
                        f"Improvement detected! Best {monitor} so far: {best_metric:.4f}"
                    )

                    if restore_best_weights:
                        best_state_dict = {
                            k: v.cpu().clone()
                            for k, v in pred_net.model.state_dict().items()
                        }
                else:
                    wait += 1
                    if wait >= patience:
                        logger.info(
                            f"\nEarly stopping triggered! No improvement in {patience} epochs."
                        )
                        logger.info(
                            f"Best {monitor} = {best_metric:.4f} at epoch {best_epoch+1}"
                        )

                        if restore_best_weights and best_state_dict:
                            pred_net.model.load_state_dict(
                                {
                                    k: v.to(self.device)
                                    for k, v in best_state_dict.items()
                                }
                            )
                            logger.info(
                                f"Restored model weights from epoch {best_epoch+1}"
                            )

                        break

            os.makedirs("weights", exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            weight_path = f"weights/{predicate_name_lower}_predicate_weights.pth"
            log_path = f"{log_dir}/{predicate_name_lower}_training_log_{timestamp}.json"

            torch.save(pred_net.model.state_dict(), weight_path)

            metadata = {
                "training_metadata": {
                    "early_stopping": {
                        "enabled": early_stopping,
                        "monitor": monitor,
                        "min_delta": min_delta,
                        "patience": patience,
                        "best_epoch": best_epoch + 1,
                        "best_metric": float(best_metric),
                        "stopped_epoch": epoch + 1,
                    },
                    "lr_scheduler": {
                        "enabled": lr_scheduler,
                        "factor": lr_factor,
                        "patience": lr_patience,
                        "min_lr": lr_min,
                        "initial_lr": lr,
                        "final_lr": current_lr,
                    },
                }
            }
            training_log.append(metadata)

            with open(log_path, "w") as f:
                json.dump(training_log, f, indent=4)
            logger.info(f"Saved {predicate_name} predicate weights to '{weight_path}'")
            logger.info(f"Saved training log to '{log_path}'")

            logger.removeHandler(file_handler)

    def inference(
        self,
        subj_class: str,
        obj_class: str,
        predicate: str,
        threshold=0.7,
        lambda_val=10.0,
    ) -> dict:
        """Perform inference using a predicate network with soft aggregation.

        Args:
            subj_class (str): Class name of the subject.
            obj_class (str): Class name of the object.
            predicate (str): Predicate to evaluate (e.g., "in", "on", "next to").
            threshold (float, optional): Confidence threshold for determining existence. Defaults to 0.7.
            lambda_val (float, optional): Temperature parameter for soft aggregation. Higher values make aggregation more strict. Defaults to 10.0.

        Returns:
            dict: A dictionary containing:
                - exists (bool): True if the relationship exists with confidence above the threshold.
                - confidence (float): The aggregated confidence score.
                - soft_confidence (float): Soft aggregation confidence score.
                - message (str): Status message of the inference.
                - subject_locations (dict): Locations of detected subject objects.
                - object_locations (dict): Locations of detected object objects.
                - subject_class (str): The subject class.
                - object_class (str): The object class.
                - predicate (str): The evaluated predicate.
                - individual_scores (list): Scores for each subject-object pair.
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

            # Traditional min aggregation (universal quantification)
            min_score = torch.min(preds).item()

            # Soft exists implementation (using log-sum-exp for differentiable soft max)
            # Formula: (1/λ) · log(∑ exp(λ · (Prédicat(s, o) - threshold))) > 0
            diff = preds - threshold
            exp_terms = torch.exp(lambda_val * diff)
            sum_exp = torch.sum(exp_terms)
            soft_score = (1.0 / lambda_val) * torch.log(sum_exp)
            soft_exists = soft_score > 0

            # Traditional max aggregation (existential quantification)
            max_score = torch.max(preds).item()

            # Reshape predictions to subject × object matrix for individual scores
            individual_scores = preds.reshape(n, m).cpu().tolist()

            return {
                "exists": soft_exists.item(),  # Use soft exists as primary decision
                "min_score": round(
                    min_score, 3
                ),  # Universal quantification (all pairs)
                "max_score": round(
                    max_score, 3
                ),  # Existential quantification (at least one pair)
                "soft_score": round(soft_score.item(), 3),  # Soft exists score
                "confidence": round(
                    soft_score.item(), 3
                ),  # Use soft score as confidence
                "message": "Inference successful",
                "subject_locations": subj_locations,
                "object_locations": obj_locations,
                "subject_class": subj_class,
                "object_class": obj_class,
                "predicate": predicate,
                "individual_scores": individual_scores,
                "threshold": threshold,
                "lambda": lambda_val,
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
