import torch
import torch.nn as nn
import ltn
import tomllib
from models.Relationship_Predicate import In, On, NextTo, OnTopOf, Near, Under
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load configuration
with open("config.toml", "rb") as f:
    config = tomllib.load(f)
ltn_config = config["Logic_Tensor_Networks"]

class Logic_Tensor_Networks:
    def __init__(self, detector_output: dict, input_dim: int, class_labels: list):
        """
        Initializes the Logic Tensor Networks with the given detector output and configuration.
        The expected detector_output keys are "centers", "widths", "heights", and "classes".
        The input_dim should be 5 (center_x, center_y, width, height, class).
        """
        self.detector_output = detector_output
        self.class_labels = class_labels

        # Build LTN variables from the detector output
        self.variables = self._variable_builder(detector_output)

        # Initialize relationship predicate networks
        self.in_predicate       = ltn.Predicate(In(input_dim))
        self.on_predicate       = ltn.Predicate(On(input_dim))
        self.next_to_predicate  = ltn.Predicate(NextTo(input_dim))
        self.on_top_of_predicate = ltn.Predicate(OnTopOf(input_dim))
        self.near_predicate     = ltn.Predicate(Near(input_dim))
        self.under_predicate    = ltn.Predicate(Under(input_dim))

        # Initialize logical connectives and quantifiers
        self.And     = ltn.Connective(ltn.fuzzy_ops.AndMin())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.Forall  = ltn.Quantifier(ltn.fuzzy_ops.AggregMin(), quantifier="f")

    def _variable_builder(self, detector_output: dict):
        """
        Converts the detector_output dictionary into a dictionary of LTN Variables.
        Expected keys: "centers", "widths", "heights", "classes".
        """
        variables = {}
        required_keys = ["centers", "widths", "heights", "classes"]
        for key in required_keys:
            if key not in detector_output:
                raise ValueError(f"Missing key '{key}' in detector_output.")
            value = detector_output[key]
            if isinstance(value, list):
                tensor = [torch.tensor(item, dtype=torch.float) if not isinstance(item, torch.Tensor) 
                          else item.float() for item in value]
                concatenated = torch.stack(tensor, dim=0)
            else:
                concatenated = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float)
            if key in ["centers", "widths", "heights"] and concatenated.dim() == 1:
                concatenated = concatenated.unsqueeze(1)
            variables[key] = ltn.Variable(key, concatenated)
        print("Constructed Variables:")
        for key, var in variables.items():
            print(f"  {key}: {var.value.shape}")
        return variables

    def train_predicate(self, predicate_name: str, train_data: Dataset, epochs: int, batch_size: int, lr: float):
        """
        Trains a single relationship predicate network using a DataLoader.
        
        :param predicate_name: One of "in", "on", "next_to", "on_top_of", "near", "under"
        :param train_data: A PyTorch Dataset returning (subject_features, object_features, label)
                           where features have shape [input_dim] and label is 1 for positive and 0 for negative samples.
        :param epochs: Number of training epochs.
        :param batch_size: Mini-batch size.
        :param lr: Learning rate.
        """
        predicate_name_lower = predicate_name.lower()
        if predicate_name_lower == "in":
            pred_net = self.in_predicate
        elif predicate_name_lower == "on":
            pred_net = self.on_predicate
        elif predicate_name_lower == "next_to":
            pred_net = self.next_to_predicate
        elif predicate_name_lower == "on_top_of":
            pred_net = self.on_top_of_predicate
        elif predicate_name_lower == "near":
            pred_net = self.near_predicate
        elif predicate_name_lower == "under":
            pred_net = self.under_predicate
        else:
            raise ValueError(f"Invalid predicate: {predicate_name}")

        pred_net.train()
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(pred_net.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                subj_features, obj_features, labels = batch
                labels = labels.view(-1, 1).float()

                # Wrap the inputs as LTN Variables using Variable (or Variable if needed)
                subj_obj = ltn.Variable("subj", subj_features)  # shape: [16,5]
                obj_obj = ltn.Variable("obj", obj_features)

                # Call the predicate network with a tuple of LTN objects
                outputs = pred_net(subj_obj, obj_obj).value.diag().unsqueeze(1)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} for predicate '{predicate_name}', Loss: {avg_loss:.4f}")

        import os
        os.makedirs("weights", exist_ok=True)
        weight_path = f"weights/{predicate_name_lower}_predicate_weights.pth"
        torch.save(pred_net.state_dict(), weight_path)
        print(f"Saved {predicate_name} predicate weights to '{weight_path}'.")

    def inference(self, subj_class: str, obj_class: str, predicate: str, threshold=0.7) -> dict:
        """
        简化版关系推理接口
        返回: {
            "exists": bool,     # 是否存在符合条件的关系
            "confidence": float, # 最高置信度
            "message": str      # 执行状态描述
        }
        """
        try:
            # 验证类别有效性
            subj_id = self.class_labels.index(subj_class)
            obj_id = self.class_labels.index(obj_class)
            
            # 筛选对象
            subj_mask = self.detector_output["classes"] == subj_id
            obj_mask = self.detector_output["classes"] == obj_id
            
            if not subj_mask.any():
                return {"exists": False, "confidence": 0.0, "message": f"未检测到 {subj_class}"}
            if not obj_mask.any():
                return {"exists": False, "confidence": 0.0, "message": f"未检测到 {obj_class}"}

            # 构建特征对
            subj_features = self._build_features(subj_mask)
            obj_features = self._build_features(obj_mask)
            
            # 批量推理
            scores = []
            for subj in subj_features:
                for obj in obj_features:
                    scores.append(self._eval_predicate(subj, obj, predicate))
            
            max_score = max(scores) if scores else 0.0
            return {
                "exists": max_score >= threshold,
                "confidence": round(max_score, 3),
                "message": "推理成功" if scores else "无有效对象对"
            }

        except ValueError as e:
            return {"exists": False, "confidence": 0.0, "message": str(e)}

    def _build_features(self, mask: torch.Tensor) -> torch.Tensor:
        """从掩码构建特征矩阵"""
        return torch.cat([
            self.detector_output["centers"][mask],
            self.detector_output["widths"][mask].unsqueeze(1),
            self.detector_output["heights"][mask].unsqueeze(1),
            self.detector_output["classes"][mask].unsqueeze(1).float()
        ], dim=1)

    def _eval_predicate(self, subj: torch.Tensor, obj: torch.Tensor, predicate: str) -> float:
        """单对对象关系评估"""
        return self.inference(
            subj.unsqueeze(0), 
            obj.unsqueeze(0), 
            predicate
        ).item()

if __name__ == "__main__":
    # Example usage in train.py
    import json

    with open("sample_dataset.json", "r") as f:
        raw_data = json.load(f)

    pos_predicate = "ON"
    neg_predicates = ["wears", "has", "next to", "on top of", "in", "behind", "holding", "parked on", "by"]

    # Assume RelationshipDataset is defined in utils/DataLoader.py
    from utils.DataLoader import RelationshipDataset

    train_dataset = RelationshipDataset(data=raw_data, pos_predicate=pos_predicate, neg_predicates=neg_predicates)

    # Create a simulated detector_output (only required keys)
    num_obj = 10
    detector_output = {
        "centers": torch.randn(num_obj, 2),
        "widths": torch.randn(num_obj),
        "heights": torch.randn(num_obj),
        "classes": torch.randint(0, 100, (num_obj,))
    }
    class_labels = list(range(100))
    input_dim = 5  # [center_x, center_y, width, height, class]

    # Initialize the Logic Tensor Networks instance
    ltn_network = Logic_Tensor_Networks(detector_output, input_dim, class_labels)
    # Optionally set the training dataset in the instance
    ltn_network.train_dataset = train_dataset

    # Train the "ON" predicate using the DataLoader method
    ltn_network.train_predicate(predicate_name="on", train_data=train_dataset, epochs=500, batch_size=16, lr=0.001)