# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tomllib
from pathlib import Path
from models import OneFormer_Extractor, LTN
from PIL import Image
import xml.etree.ElementTree as ET
import os

############################
# 1) 在全局作用域下定义数据集类
############################
class VOCRelationDataset(Dataset):
    def __init__(self, root_dir, extractor):
        super().__init__()
        self.root_dir = root_dir
        self.extractor = extractor
        # 这里根据实际项目进行初始化
        self.image_paths = "data/VOCdevkit/VOC2012/JPEGImages"
        self.annotations = "data/VOCdevkit/VOC2012/Annotations"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 示例：加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # 转为Tensor
        image_tensor = self.extractor.transform(image)

        # 示例：关系标签
        relations = {
            "Near": [
                (0, 1, 1),
                (1, 2, 0),
            ],
            "TopOf": [
                (0, 2, 1),
            ]
        }
        return {
            "image": image_tensor,
            "relations": relations
        }

def collate_fn(batch, extractor):
    """自定义批处理函数"""
    try:
        images = torch.stack([item["image"] for item in batch])
        detections = extractor.predict_batch(images)

        # 将关系打包为字典
        all_relations = {"Near": [], "TopOf": []}
        for item in batch:
            all_relations["Near"].append(item["relations"]["Near"])
            all_relations["TopOf"].append(item["relations"]["TopOf"])

        return {
            "image": images,
            "detections": detections,
            "relations": all_relations
        }
    except Exception as e:
        print(f"批处理失败: {str(e)}")
        return None


def main():
    # 读取配置
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    BATCH_SIZE = config["Training"]["batch_size"]
    LR = config["Training"]["lr"]
    EPOCHS = config["Training"]["epochs"]
    SAVE_PATH = Path(config["Training"]["save_path"])
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # 初始化特征提取器
    extractor = OneFormer_Extractor()

    # 初始化数据集和DataLoader
    dataset = VOCRelationDataset(
        root_dir="data/VOCdevkit/VOC2012",
        extractor=extractor
    )

    # 注意要传入collate_fn时，把extractor作为闭包变量
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, extractor),
        num_workers=0,
        pin_memory=True
    )

    # 动态计算特征维度
    with torch.no_grad():
        sample_batch = None
        while True:
            sample_batch = next(iter(loader))
            if (sample_batch is not None and 
                sample_batch["detections"] is not None and
                "boxes" in sample_batch["detections"] and
                sample_batch["detections"]["boxes"].shape[1] > 0):
                break

        box_dim = sample_batch["detections"]["boxes"].shape[-1]
        score_dim = sample_batch["detections"]["scores"].shape[-1]
        class_dim = len(extractor.labels)
        mask_dim = 2
        feat_dim = box_dim + score_dim + class_dim + mask_dim

    # 初始化LTN
    ltn = LTN(
        detector_output=sample_batch["detections"],
        label_mapping=extractor.labels,
        config={
            "mlp_input_dim": feat_dim * 2,
            "cnn_input_dim": feat_dim * 2
        }
    )

    # 冻结提取器
    for name, param in extractor.model.named_parameters():
        param.requires_grad = False

    # 优化器、损失函数
    optimizer = optim.AdamW([
        {'params': ltn.predicates.Near.parameters(), 'lr': LR},
        {'params': ltn.predicates.TopOf.parameters(), 'lr': LR * 0.5}
    ], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))

    # 训练循环
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue

            optimizer.zero_grad()

            try:
                ltn.objects = ltn._process_detector_output(batch["detections"])

                loss = 0.0
                for rel_name in ["Near", "TopOf"]:
                    pred_scores = []
                    true_labels = []

                    batch_size_current = len(batch["relations"][rel_name])
                    for b in range(batch_size_current):
                        relations = batch["relations"][rel_name][b]
                        for (i, j, label) in relations:
                            if i >= ltn.objects.value.shape[1] or j >= ltn.objects.value.shape[1]:
                                continue
                            pair_feat = torch.cat([
                                ltn.objects.value[b, i],
                                ltn.objects.value[b, j]
                            ], dim=0)
                            score = ltn.predicates[rel_name](pair_feat.unsqueeze(0))
                            pred_scores.append(score)
                            true_labels.append(
                                torch.tensor([[label]], device=score.device).float()
                            )

                    if len(pred_scores) > 0:
                        pred = torch.cat(pred_scores, dim=0)
                        true = torch.cat(true_labels, dim=0)
                        loss += criterion(pred, true)

                if loss > 0:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            except Exception as e:
                print(f"批次 {batch_idx} 训练失败: {str(e)}")

        avg_loss = total_loss / (len(loader) if len(loader) > 0 else 1)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        # 保存模型
        if (epoch+1) % 5 == 0:
            torch.save(ltn.state_dict(), SAVE_PATH / f"ltn_epoch_{epoch+1}.pth")

    print("训练完成!")


######################
# 2) 最好使用这种入口方式
######################
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()