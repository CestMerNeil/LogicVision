import torch
import json
from torch.utils.data import Dataset
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class RelationshipDataset(Dataset):
    def __init__(self, relationships_json_path, image_meta_json_path, pos_predicate, neg_predicates, use_cuda=True):
        """
        Initialize the dataset.
        """
        self.samples = []
        self.pos_predicate = pos_predicate.lower()
        self.neg_predicates = [p.lower() for p in neg_predicates]
        
        # Choose CUDA if available
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load image metadata from JSON file
        with open(image_meta_json_path, 'r') as f:
            image_meta_data = json.load(f)
        self.image_meta = {img["image_id"]: img for img in image_meta_data}

        # Load relationships data from JSON file
        with open(relationships_json_path, 'r') as f:
            relationships_data = json.load(f)

        # Process relationship data with multithreading
        print("Processing relationship data with multithreading...")
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(self._process_image, relationships_data), total=len(relationships_data), desc="Loading Images"))

    def _process_image(self, image_entry):
        image_id = image_entry["image_id"]
        relationships = image_entry.get("relationships", [])
        if image_id not in self.image_meta:
            return
        img_width = self.image_meta[image_id]["width"]
        img_height = self.image_meta[image_id]["height"]

        positive_pairs = []
        object_features = {}

        # Process each relationship in the image
        for rel in relationships:
            subj_feat = self._extract_features(rel["subject"], img_width, img_height)
            obj_feat = self._extract_features(rel["object"], img_width, img_height)
            if subj_feat is None or obj_feat is None:
                continue

            if rel["predicate"].lower() == self.pos_predicate:
                # Append positive sample with label 1.0
                self.samples.append((subj_feat, obj_feat, torch.tensor(1.0, device=self.device)))
                subj_id = rel["subject"].get("object_id")
                obj_id = rel["object"].get("object_id")
                if subj_id and obj_id:
                    positive_pairs.append((subj_id, obj_id))

            # Store object features for later negative sampling if object_id exists
            if "object_id" in rel["subject"]:
                object_features[rel["subject"]["object_id"]] = subj_feat
            if "object_id" in rel["object"]:
                object_features[rel["object"]["object_id"]] = obj_feat

        # Generate negative samples for this image
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
            return torch.tensor([x, y, w, h, class_id], dtype=torch.float32, device=self.device)
        except (KeyError, TypeError, ValueError):
            return None

    def _generate_negative_samples(self, object_features, positive_pairs):
        """
        Efficiently generate negative samples using random sampling.
        
        Instead of iterating over all possible pairs, this method randomly samples pairs
        until we have as many negatives as positive samples (or until no more negatives are possible).
        """
        object_ids = list(object_features.keys())
        num_objects = len(object_ids)
        if num_objects < 2:
            return

        # Create a set for quick lookup of positive pairs
        positive_set = set(positive_pairs)
        # Calculate the maximum possible negative pairs
        max_possible_negatives = num_objects * (num_objects - 1) - len(positive_set)
        num_positives = len(positive_pairs)
        num_negatives_needed = min(num_positives, max_possible_negatives)
        if num_negatives_needed <= 0:
            return

        candidate_negatives = set()

        # Randomly sample negative pairs until the required number is reached
        while len(candidate_negatives) < num_negatives_needed:
            subj_id, obj_id = random.sample(object_ids, 2)
            pair = (subj_id, obj_id)
            if pair in positive_set or pair in candidate_negatives:
                continue
            candidate_negatives.add(pair)

        # Append negative samples with label 0.0
        for subj_id, obj_id in candidate_negatives:
            self.samples.append((object_features[subj_id], object_features[obj_id], torch.tensor(0.0, device=self.device)))

        # Shuffle the samples to mix positive and negative examples
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj_feat, obj_feat, label = self.samples[idx]
        return subj_feat, obj_feat, label