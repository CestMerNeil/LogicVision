import torch
import json
from torch.utils.data import Dataset
import random

class RelationshipDataset(Dataset):
    def __init__(self, relationships_json_path, image_meta_json_path, pos_predicate, neg_predicates):
        """
        Initialize the dataset.

        Args:
            relationships_json_path (str): Path to relationships.json.
            image_meta_json_path (str): Path to image_data.json.
            pos_predicate (str): Positive predicate (e.g., "ON").
            neg_predicates (list): List of negative predicates (e.g., ["has", "wears"]).
        """
        self.samples = []
        self.pos_predicate = pos_predicate.lower()
        self.neg_predicates = [p.lower() for p in neg_predicates]

        with open(image_meta_json_path, 'r') as f:
            image_meta_data = json.load(f)
        self.image_meta = {img["image_id"]: img for img in image_meta_data}

        with open(relationships_json_path, 'r') as f:
            relationships_data = json.load(f)

        for image_entry in relationships_data:
            self._process_image(image_entry)

    def _process_image(self, image_entry):
        image_id = image_entry["image_id"]
        relationships = image_entry.get("relationships", [])
        if image_id not in self.image_meta:
            return
        img_width = self.image_meta[image_id]["width"]
        img_height = self.image_meta[image_id]["height"]

        positive_pairs = []
        object_features = {}

        for rel in relationships:
            subj_feat = self._extract_features(rel["subject"], img_width, img_height)
            obj_feat = self._extract_features(rel["object"], img_width, img_height)
            if subj_feat is None or obj_feat is None:
                continue

            if rel["predicate"].lower() == self.pos_predicate:
                self.samples.append((subj_feat, obj_feat, 1.0))
                subj_id = rel["subject"].get("object_id")
                obj_id = rel["object"].get("object_id")
                if subj_id and obj_id:
                    positive_pairs.append((subj_id, obj_id))

            if "object_id" in rel["subject"]:
                object_features[rel["subject"]["object_id"]] = subj_feat
            if "object_id" in rel["object"]:
                object_features[rel["object"]["object_id"]] = obj_feat

        self._generate_negative_samples(object_features, positive_pairs)

    def _extract_features(self, obj, img_width, img_height):
        """
        Extract normalized features: [x_center, y_center, width, height, class_id].
        """
        try:
            x = obj["x"] / img_width
            y = obj["y"] / img_height
            w = obj["w"] / img_width
            h = obj["h"] / img_height
            if not (0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0):
                return None
            class_id = float(obj["object_id"]) / 1e6
            return torch.tensor([x, y, w, h, class_id], dtype=torch.float32)
        except (KeyError, TypeError, ValueError):
            return None

    def _generate_negative_samples(self, object_features, positive_pairs):
        """
        Generate negative samples.
        """
        object_ids = list(object_features.keys())
        candidate_negatives = []
        for i in range(len(object_ids)):
            for j in range(len(object_ids)):
                if i != j:
                    pair = (object_ids[i], object_ids[j])
                    if pair not in positive_pairs:
                        candidate_negatives.append(
                            (object_features[object_ids[i]], object_features[object_ids[j]])
                        )
        num_pos = len(positive_pairs)
        if candidate_negatives:
            sampled_negatives = random.sample(candidate_negatives, min(num_pos, len(candidate_negatives)))
            for neg in sampled_negatives:
                self.samples.append((neg[0], neg[1], 0.0))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj_feat, obj_feat, label = self.samples[idx]
        return subj_feat, obj_feat, torch.tensor(label, dtype=torch.float32)