# utils/Trainer.py
import tomllib
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models import LTN, OneFormer_Extractor
from utils.DataLoader import get_openimages_dataloader

def compute_supervised_near_loss(ltn_model, relationship_annotations):
    """
    对于 batch 中每张图像，根据其关系标注（只对 predicate=="Near" 进行监督）计算 BCE 损失。
    对于每个标注，提取对应对象对的特征，经 Near 谓词网络预测分数，与目标值 1 比较。
    """
    bce_loss = nn.BCELoss()
    total_loss = 0.0
    total_count = 0
    for b, annots in enumerate(relationship_annotations):
        for annot in annots:
            if annot["predicate"] != "Near":
                continue
            subj_idx = annot["subject_index"]
            obj_idx = annot["object_index"]
            # 检查下标是否在当前图像中有效（可能标注与检测对象数目不匹配）
            if subj_idx >= ltn_model.objects.value[b].shape[0] or obj_idx >= ltn_model.objects.value[b].shape[0]:
                continue
            pair_feature = torch.cat([ltn_model.objects.value[b, subj_idx],
                                        ltn_model.objects.value[b, obj_idx]], dim=0)
            pair_feature = pair_feature.unsqueeze(0)  # [1, mlp_input_dim]
            pred = ltn_model.predicates["Near"](pair_feature).view(-1)
            target = torch.ones_like(pred)
            loss = bce_loss(pred, target)
            total_loss += loss
            total_count += 1
    if total_count > 0:
        return total_loss / total_count
    else:
        return torch.tensor(0.0)

def train():
    # 读取配置文件
    config_path = Path("config.toml")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    trainer_config = config["Trainer"]

    # 由于 OneFormer 不支持 .to()，因此这里始终在 CPU 上运行
    device = torch.device("cpu")
    print("Using device:", device)

    # 获取 OpenImages 数据加载器
    train_loader = get_openimages_dataloader()

    # 初始化 OneFormer 特征提取器（固定，不更新，也不调用 .to()）
    feature_extractor = OneFormer_Extractor()
    # feature_extractor.model.eval() 依然可以调用，但不调用 .to()

    # 构造一个“虚拟”检测输出用于初始化 LTN 模型（格式需与 OneFormer 输出一致）
    dummy_detector_output = {
        'boxes': torch.zeros(1, 10, 4),
        'classes': torch.zeros(1, 10, dtype=torch.long),
        'masks': torch.zeros(1, 10, 640, 640),
        'scores': torch.zeros(1, 10),
        'image_size': torch.tensor([[640, 640]], dtype=torch.long)
    }
    # 定义类别映射（需要与数据集中的类别及标注保持一致）
    labels = {0: 'background', 1: 'cup', 2: 'bottle', 3: 'table', 4: 'chair'}

    # 初始化 LTN 模型（仅含 Near 谓词网络）
    ltn_model = LTN(dummy_detector_output, labels)
    ltn_model.predicates.to(device)

    optimizer = optim.Adam(
        ltn_model.predicates.parameters(),
        lr = trainer_config["init_lr"],
        weight_decay = trainer_config["weight_decay"]
    )
    epochs = trainer_config["epochs"]
    grad_clip = trainer_config["grad_clip"]
    output_dir = Path(trainer_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ltn_model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch_idx, (images, relationship_annotations) in enumerate(train_loader):
            optimizer.zero_grad()

            # 利用 OneFormer 提取器获得检测输出（批量预测）
            with torch.no_grad():
                detector_output = feature_extractor.predict_batch(images)
            # 更新 LTN 模型中检测数据与对象特征
            ltn_model.raw_data = detector_output
            ltn_model.objects = ltn_model._process_detector_output(detector_output)

            loss = compute_supervised_near_loss(ltn_model, relationship_annotations)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ltn_model.predicates.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % trainer_config["save_interval"] == 0:
            save_path = output_dir / f"Near_epoch{epoch+1}.pth"
            torch.save(ltn_model.predicates.state_dict(), save_path)
            print(f"Saved model weights to {save_path}")

    final_save_path = output_dir / "Near_final.pth"
    torch.save(ltn_model.predicates.state_dict(), final_save_path)
    print(f"Training completed. Final model weights saved to {final_save_path}")

if __name__ == "__main__":
    train()