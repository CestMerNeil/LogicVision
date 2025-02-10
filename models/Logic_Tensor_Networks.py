import torch
import torch.nn as nn
import ltn
import tomllib
from models.Relationship_Predicate import In, On, NextTo, OnTopOf, Near, Under
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def auto_select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class Logic_Tensor_Networks:
    def __init__(self, detector_output: dict, input_dim: int, class_labels: list, device: torch.device = None):
        """
        Initializes the Logic Tensor Networks with the given detector output and configuration.
        The expected detector_output keys are "centers", "widths", "heights", and "classes".
        The input_dim should be 5 (center_x, center_y, width, height, class).
        """
        if device is None:
            device = auto_select_device()
        self.device = device
        print(f"Using device: {self.device}")

        processed_detector_output = {}
        for key, value in detector_output.items():
            if isinstance(value, torch.Tensor):
                processed_detector_output[key] = value.to(self.device)
            elif isinstance(value, list):
                tensor_list = [
                    torch.tensor(item, dtype=torch.float, device=self.device)
                    if not isinstance(item, torch.Tensor) else item.to(self.device)
                    for item in value
                ]
                processed_detector_output[key] = torch.stack(tensor_list, dim=0)
            else:
                processed_detector_output[key] = value
        self.detector_output = processed_detector_output
        self.class_labels = class_labels

        # Build LTN variables from the detector output
        self.variables = self._variable_builder(self.detector_output)

        # Initialize relationship predicate networks
        self.in_predicate        = ltn.Predicate(In(input_dim)).to(self.device)
        self.on_predicate        = ltn.Predicate(On(input_dim)).to(self.device)
        self.next_to_predicate   = ltn.Predicate(NextTo(input_dim)).to(self.device)
        self.on_top_of_predicate = ltn.Predicate(OnTopOf(input_dim)).to(self.device)
        self.near_predicate      = ltn.Predicate(Near(input_dim)).to(self.device)
        self.under_predicate     = ltn.Predicate(Under(input_dim)).to(self.device)

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
                tensor = [
                    torch.tensor(item, dtype=torch.float, device=self.device)
                    if not isinstance(item, torch.Tensor) else item.to(self.device)
                    for item in value
                ]
                concatenated = torch.stack(tensor, dim=0)
            else:
                concatenated = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float, device=self.device)
            if key in ["centers", "widths", "heights"] and concatenated.dim() == 1:
                concatenated = concatenated.unsqueeze(1)
            variables[key] = ltn.Variable(key, concatenated)
        print("Constructed Variables:")
        for key, var in variables.items():
            print(f"  {key}: {var.value.shape}")
        return variables

    def train_predicate(self, predicate_name: str, full_data: Dataset, epochs: int, batch_size: int, lr: float, val_split: float = 0.2):
        """
        使用单一数据集自动分割成训练集和验证集，对指定关系谓词进行训练和验证。

        :param predicate_name: 关系谓词名称，取值为 "in", "on", "next_to", "on_top_of", "near", "under" 之一。
        :param full_data: 一个 PyTorch Dataset，返回 (subject_features, object_features, label)
                        其中 features 的形状为 [input_dim]，label 为 1（正例）或 0（负例）。
        :param epochs: 训练的 epoch 数量。
        :param batch_size: mini-batch 大小。
        :param lr: 学习率。
        :param val_split: 验证集占总数据集的比例，默认 0.2（即 20% 用于验证）。
        """
        # 仅在此处判断一次谓词名称，选择对应的谓词网络
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

        # 自动分割数据集：训练集与验证集
        total_size = len(full_data)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_data, [train_size, val_size])
        print(f"Total samples: {total_size}, Training samples: {train_size}, Validation samples: {val_size}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(pred_net.parameters(), lr=lr)

        # 开始训练和验证循环
        for epoch in range(epochs):
            pred_net.train()  # 训练模式
            epoch_train_loss = 0.0
            train_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False):
                subj_features, obj_features, labels = batch
                subj_features = subj_features.to(self.device)
                obj_features = obj_features.to(self.device)
                labels = labels.view(-1, 1).float().to(self.device)

                # 将输入包装为 LTN Variable（确保数据在 self.device 上）
                subj_obj = ltn.Variable("subj", subj_features)
                obj_obj = ltn.Variable("obj", obj_features)

                # 前向传播，并提取对角线上的输出作为预测值
                outputs = pred_net(subj_obj, obj_obj).value.diag().unsqueeze(1)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                train_batches += 1

            avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0.0

            # 验证阶段：模型切换到评估模式，且不计算梯度
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

                    outputs = pred_net(subj_obj, obj_obj).value.diag().unsqueeze(1)
                    loss = loss_fn(outputs, labels)
                    epoch_val_loss += loss.item()
                    val_batches += 1

                    # 使用 0.5 为阈值判断预测类别
                    preds = (outputs >= 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else 0.0
            accuracy = correct / total if total > 0 else 0.0

            # 每个 epoch 输出训练与验证结果
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, "
                    f"Val Loss = {avg_val_loss:.4f}, Val Accuracy = {accuracy:.4f}")

        # 保存训练后的权重
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
        features = torch.cat([
            self.detector_output["centers"][mask],
            self.detector_output["widths"][mask].unsqueeze(1),
            self.detector_output["heights"][mask].unsqueeze(1),
            self.detector_output["classes"][mask].unsqueeze(1).float()
        ], dim=1)
        return features.to(self.device)

if __name__ == "__main__":
    # Example usage in train.py
    import json

    pos_predicate = "ON"
    neg_predicates = ["wears", "has", "next to", "on top of", "in", "behind", "holding", "parked on", "by"]

    # Assume RelationshipDataset is defined in utils/DataLoader.py
    from utils.DataLoader import RelationshipDataset

    train_dataset = RelationshipDataset(
        relationships_json_path="data/relationships.json",
        image_meta_json_path="data/image_data.json",
        pos_predicate=pos_predicate, 
        neg_predicates=neg_predicates
    )

    # 创建一个模拟的 detector_output（仅包含必须的键）
    num_obj = 10
    device = auto_select_device()
    detector_output = {
        "centers": torch.randn(num_obj, 2, device=device),
        "widths": torch.randn(num_obj, device=device),
        "heights": torch.randn(num_obj, device=device),
        "classes": torch.randint(0, 100, (num_obj,), device=device)
    }
    class_labels = list(range(100))
    input_dim = 5  # [center_x, center_y, width, height, class]

    # Initialize the Logic Tensor Networks instance
    ltn_network = Logic_Tensor_Networks(detector_output, input_dim, class_labels, device=device)
    ltn_network.train_dataset = train_dataset

    # Train the "ON" predicate using the DataLoader method
    ltn_network.train_predicate(predicate_name="on", train_data=train_dataset, epochs=100, batch_size=1024, lr=0.001)