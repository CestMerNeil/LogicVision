import torch
import json
from torch.utils.data import Dataset
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class RelationshipDataset(Dataset):
    def __init__(self, relationships_json_path, image_meta_json_path, pos_predicate, neg_predicates, use_cuda=True):
        self.samples = []
        self.pos_predicate = pos_predicate.lower()
        self.neg_predicates = [p.lower() for p in neg_predicates]
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 使用生成器逐块加载JSON
        def chunked_json_loader(path, chunk_size=1000):
            with open(path, 'r') as f:
                data = json.load(f)
                for i in range(0, len(data), chunk_size):
                    yield data[i:i+chunk_size]

        # 加载图像元数据
        with open(image_meta_json_path, 'r') as f:
            self.image_meta = {img["image_id"]: img for img in json.load(f)}

        # 多线程处理优化
        print("Optimizing data loading with parallel processing...")
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for chunk in chunked_json_loader(relationships_json_path):
                futures.append(executor.submit(self.process_chunk, chunk))
            
            for future in tqdm(futures, desc="Processing chunks"):
                chunk_samples = future.result()
                self.samples.extend(chunk_samples)

        # 最终全局shuffle
        random.shuffle(self.samples)

    def process_chunk(self, chunk):
        chunk_samples = []
        for image_entry in chunk:
            chunk_samples.extend(self._process_image(image_entry))
        return chunk_samples

    def _process_image(self, image_entry):
        image_id = image_entry["image_id"]
        if image_id not in self.image_meta:
            return []
        
        img_meta = self.image_meta[image_id]
        img_width, img_height = img_meta["width"], img_meta["height"]
        object_features = {}
        positive_pairs = set()
        samples = []

        # 批量特征提取
        for rel in image_entry.get("relationships", []):
            # 处理subject
            subj_feat = self._extract_features(rel["subject"], img_width, img_height)
            if subj_feat is not None and "object_id" in rel["subject"]:
                object_features[rel["subject"]["object_id"]] = subj_feat
            
            # 处理object
            obj_feat = self._extract_features(rel["object"], img_width, img_height)
            if obj_feat is not None and "object_id" in rel["object"]:
                object_features[rel["object"]["object_id"]] = obj_feat

            if subj_feat is not None and obj_feat is not None:
                predicate = rel["predicate"].lower()
                if predicate == self.pos_predicate:
                    samples.append((subj_feat, obj_feat, torch.tensor(1.0)))
                    if "object_id" in rel["subject"] and "object_id" in rel["object"]:
                        positive_pairs.add((rel["subject"]["object_id"], rel["object"]["object_id"]))

        # 批量生成负样本
        if len(object_features) >= 2:
            obj_ids = list(object_features.keys())
            obj_features = list(object_features.values())
            
            # 向量化生成候选对
            num_neg = min(len(positive_pairs), len(obj_ids)*(len(obj_ids)-1)//2)
            if num_neg > 0:
                # 使用矩阵运算生成所有可能组合
                indices = np.array(np.meshgrid(np.arange(len(obj_ids)), np.arange(len(obj_ids)))).T.reshape(-1,2)
                mask = (indices[:,0] != indices[:,1])
                candidate_pairs = indices[mask]

                # 转换为ID对并过滤正样本
                candidate_ids = [(obj_ids[i], obj_ids[j]) for i,j in candidate_pairs]
                valid_mask = [pair not in positive_pairs for pair in candidate_ids]
                valid_pairs = np.array(candidate_ids)[valid_mask]

                # 随机选择
                if len(valid_pairs) > num_neg:
                    selected = valid_pairs[np.random.choice(len(valid_pairs), num_neg, replace=False)]
                else:
                    selected = valid_pairs

                # 添加负样本
                for s_id, o_id in selected:
                    s_idx = obj_ids.index(s_id)
                    o_idx = obj_ids.index(o_id)
                    samples.append((obj_features[s_idx], obj_features[o_idx], torch.tensor(0.0)))

        return samples

    def _extract_features(self, obj, img_width, img_height):
        """向量化特征提取"""
        try:
            x = obj["x"] / img_width
            y = obj["y"] / img_height
            w = obj["w"] / img_width
            h = obj["h"] / img_height
            if not (0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0):
                return None
            class_id = float(obj.get("object_id", 0)) % 1000  # 限制类别数量
            return torch.tensor([x, y, w, h, class_id], dtype=torch.float32)
        except (KeyError, TypeError, ValueError):
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj_feat, obj_feat, label = self.samples[idx]
        return subj_feat, obj_feat, label