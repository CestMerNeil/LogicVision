# utils/DataLoader.py
import torch
import json
from torch.utils.data import Dataset
import random

class RelationshipDataset(Dataset):
    def __init__(self, relationships_json_path, image_meta_json_path, pos_predicate, neg_predicates):
        """
        :param relationships_json_path: relationships.json 路径
        :param image_meta_json_path: image_data.json 路径
        :param pos_predicate: 目标正样本谓词（如 "ON"）
        :param neg_predicates: 负样本谓词列表（如 ["has", "wears"]）
        """
        self.samples = []
        self.pos_predicate = pos_predicate.lower()
        self.neg_predicates = [p.lower() for p in neg_predicates]

        # 加载图像元数据并建立索引
        with open(image_meta_json_path, 'r') as f:
            image_meta_data = json.load(f)
        self.image_meta = {img["image_id"]: img for img in image_meta_data}

        # 加载关系数据
        with open(relationships_json_path, 'r') as f:
            relationships_data = json.load(f)

        # 处理每张图像的关系
        for image_entry in relationships_data:
            self._process_image(image_entry)

    def _process_image(self, image_entry):
        image_id = image_entry["image_id"]
        relationships = image_entry.get("relationships", [])
        
        # 获取图像尺寸用于归一化
        if image_id not in self.image_meta:
            return  # 跳过缺失元数据的图像
        img_width = self.image_meta[image_id]["width"]
        img_height = self.image_meta[image_id]["height"]

        positive_pairs = []
        object_features = {}

        # 处理正样本
        for rel in relationships:
            # 提取subject和object特征（带归一化）
            subj_feat = self._extract_features(rel["subject"], img_width, img_height)
            obj_feat = self._extract_features(rel["object"], img_width, img_height)
            
            if subj_feat is None or obj_feat is None:
                continue

            # 记录正样本对
            if rel["predicate"].lower() == self.pos_predicate:
                self.samples.append((subj_feat, obj_feat, 1.0))
                subj_id = rel["subject"].get("object_id")
                obj_id = rel["object"].get("object_id")
                if subj_id and obj_id:
                    positive_pairs.append((subj_id, obj_id))

            # 记录所有对象特征
            if "object_id" in rel["subject"]:
                object_features[rel["subject"]["object_id"]] = subj_feat
            if "object_id" in rel["object"]:
                object_features[rel["object"]["object_id"]] = obj_feat

        # 生成负样本
        self._generate_negative_samples(object_features, positive_pairs)

    def _extract_features(self, obj, img_width, img_height):
        """提取归一化后的特征 [x_center, y_center, width, height, class_id]"""
        try:
            # 归一化坐标和尺寸到 [0,1]
            x = obj["x"] / img_width
            y = obj["y"] / img_height
            w = obj["w"] / img_width
            h = obj["h"] / img_height
            
            # 验证数值范围
            if not (0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0):
                return None

            # 使用object_id作为类别标识（需确保唯一性）
            class_id = float(obj["object_id"]) / 1e6  # 假设object_id范围在1e6以内

            return torch.tensor([x, y, w, h, class_id], dtype=torch.float32)
        except (KeyError, TypeError, ValueError):
            return None

    def _generate_negative_samples(self, object_features, positive_pairs):
        """生成负样本逻辑"""
        object_ids = list(object_features.keys())
        candidate_negatives = []

        # 生成所有可能的不在正样本中的对象对
        for i in range(len(object_ids)):
            for j in range(len(object_ids)):
                if i != j:
                    pair = (object_ids[i], object_ids[j])
                    if pair not in positive_pairs:
                        candidate_negatives.append(
                            (object_features[object_ids[i]], object_features[object_ids[j]])
                        )

        # 采样与正样本相同数量的负样本
        num_pos = len(positive_pairs)
        if candidate_negatives:
            sampled_negatives = random.sample(
                candidate_negatives, 
                min(num_pos, len(candidate_negatives))
            )
            for neg in sampled_negatives:
                self.samples.append((neg[0], neg[1], 0.0))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj_feat, obj_feat, label = self.samples[idx]
        return subj_feat, obj_feat, torch.tensor(label, dtype=torch.float32)