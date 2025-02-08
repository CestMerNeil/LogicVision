import torch
import torch.nn as nn
import ltn
import tomllib
# from utils.NetworkBuilder import _build_cnn, _build_mlp
from Relationship_Predicate import In, On, NextTo, OnTopOf, Near, Under

with open("config.toml", "rb") as f:
    config = tomllib.load(f)
ltn_config = config["Logic_Tensor_Networks"]

class Logic_Tensor_Networks:
    def __init__(self, detector_output: dict, input_dim: int, class_labels: list):
        self.detector_output = detector_output
        self.class_labels = class_labels

        self.variables = self._variable_builder(detector_output)

        self.in_predicate = ltn.Predicate(In(input_dim))
        self.on_predicate = ltn.Predicate(On(input_dim))
        self.next_to_predicate = ltn.Predicate(NextTo(input_dim))
        self.on_top_of_predicate = ltn.Predicate(OnTopOf(input_dim))
        self.near_predicate = ltn.Predicate(Near(input_dim))
        self.under_predicate = ltn.Predicate(Under(input_dim))

        self.And = ltn.Connective(ltn.fuzzy_ops.And_Min())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregMin(), quantifier="f")

    def _variable_builder(self, detector_output: dict):
        variables = {}
        for key, value in detector_output.items():
            if key == "masks":
                continue
            if isinstance(value, list):
                tensors = [
                    torch.tensor(item) if not isinstance(item, torch.Tensor) else item
                ]
                concatenated = torch.stack(tensors, dim=0)
            else:
                concatenated = value if isinstance(value, torch.Tensor) else torch.tensor(value)

            variables[key] = ltn.Variable(concatenated)
        print("Variables:")
        print(variables)
        return variables

    def train_predicate(self, predicate: str, train_data: dict, epochs: int, batch_size: int, lr: float):
        if predicate.lower() == "in":
            predicate = self.in_predicate
        elif predicate.lower() == "on":
            predicate = self.on_predicate
        elif predicate.lower() == "next_to":
            predicate = self.next_to_predicate
        elif predicate.lower() == "on_top_of":
            predicate = self.on_top_of_predicate
        elif predicate.lower() == "near":
            predicate = self.near_predicate
        elif predicate.lower() == "under":
            predicate = self.under_predicate
        else:
            raise ValueError(f"Invalid predicate: {predicate}")

        # 使用内置 BCE 损失函数
        loss_fn = nn.BCELoss()

        # 优化器只更新该谓词网络的参数
        optimizer = torch.optim.Adam(pred_net.module.parameters(), lr=lr)

        pos_subj = train_data["pos"]["subject_features"]
        pos_obj  = train_data["pos"]["object_features"]
        neg_subj = train_data["neg"]["subject_features"]
        neg_obj  = train_data["neg"]["object_features"]

        N_pos = pos_subj.shape[0]
        N_neg = neg_subj.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = max((N_pos + batch_size - 1) // batch_size, (N_neg + batch_size - 1) // batch_size)
            pos_perm = torch.randperm(N_pos)
            neg_perm = torch.randperm(N_neg)

            for i in range(num_batches):
                # 划分正样本 mini-batch
                pos_start = i * batch_size
                pos_end = min((i+1) * batch_size, N_pos)
                if pos_start >= N_pos:
                    pos_batch_subj = torch.empty(0, pos_subj.shape[1])
                    pos_batch_obj  = torch.empty(0, pos_obj.shape[1])
                else:
                    pos_indices = pos_perm[pos_start:pos_end]
                    pos_batch_subj = pos_subj[pos_indices]
                    pos_batch_obj  = pos_obj[pos_indices]

                # 划分负样本 mini-batch
                neg_start = i * batch_size
                neg_end = min((i+1) * batch_size, N_neg)
                if neg_start >= N_neg:
                    neg_batch_subj = torch.empty(0, neg_subj.shape[1])
                    neg_batch_obj  = torch.empty(0, neg_obj.shape[1])
                else:
                    neg_indices = neg_perm[neg_start:neg_end]
                    neg_batch_subj = neg_subj[neg_indices]
                    neg_batch_obj  = neg_obj[neg_indices]

                # 前向传播：计算正样本和负样本的关系得分
                if pos_batch_subj.shape[0] > 0:
                    pos_score = pred_net(pos_batch_subj, pos_batch_obj)  # 输出形状 [B_pos, 1]
                else:
                    pos_score = torch.tensor([]).to(pos_subj.device)

                if neg_batch_subj.shape[0] > 0:
                    neg_score = pred_net(neg_batch_subj, neg_batch_obj)  # 输出形状 [B_neg, 1]
                else:
                    neg_score = torch.tensor([]).to(neg_subj.device)

                # 构造目标标签
                if pos_score.numel() > 0:
                    pos_labels = torch.ones_like(pos_score)
                    loss_pos = loss_fn(pos_score, pos_labels)
                else:
                    loss_pos = 0.0

                if neg_score.numel() > 0:
                    neg_labels = torch.zeros_like(neg_score)
                    loss_neg = loss_fn(neg_score, neg_labels)
                else:
                    loss_neg = 0.0

                loss = loss_pos + loss_neg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} for predicate '{predicate_name}', Loss: {avg_loss:.4f}")

    def inference(self, subject: str, object: str, predicate: str):
        if predicate.lower() == "in":
            predicate = self.in_predicate
        elif predicate.lower() == "on":
            predicate = self.on_predicate
        elif predicate.lower() == "next_to":
            predicate = self.next_to_predicate
        elif predicate.lower() == "on_top_of":
            predicate = self.on_top_of_predicate
        elif predicate.lower() == "near":
            predicate = self.near_predicate
        elif predicate.lower() == "under":
            predicate = self.under_predicate
        else:
            raise ValueError(f"Invalid predicate: {predicate}")
        
if __name__ == __main__():
    pass