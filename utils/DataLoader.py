import json
import random
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class RelationshipDataset(Dataset):
    """A PyTorch dataset for image relationship data with performance optimizations."""

    def __init__(
        self,
        relationships_json_path,
        image_meta_json_path,
        pos_predicate,
        cache_dir=None,
        use_cache=True,
        num_workers=None,
    ):
        """Initializes the RelationshipDataset and processes the data with optimized performance."""
        self.samples = []
        self.pos_predicate = pos_predicate.lower()
        
        # Create cache filename based on inputs
        if cache_dir is not None and use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = f"{os.path.basename(relationships_json_path)}_{os.path.basename(image_meta_json_path)}_{pos_predicate.replace(' ', '_')}.pkl"
            cache_path = os.path.join(cache_dir, cache_name)
            
            # Try loading from cache first
            if os.path.exists(cache_path):
                print(f"Loading cached dataset from {cache_path}")
                with open(cache_path, 'rb') as f:
                    self.samples = pickle.load(f)
                    random.shuffle(self.samples)
                    return

        # If not using cache or cache doesn't exist, process data
        print("Processing dataset from source files...")
        
        # Load image metadata once
        with open(image_meta_json_path, "r") as f:
            self.image_meta = {img["image_id"]: img for img in json.load(f)}

        # Use ProcessPoolExecutor for CPU-bound tasks instead of ThreadPoolExecutor
        num_workers = num_workers or os.cpu_count()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Load all data first (since we need image_meta in the worker processes)
            with open(relationships_json_path, "r") as f:
                all_data = json.load(f)
            
            # Calculate optimal chunk size based on data size and number of workers
            chunk_size = max(1, len(all_data) // (num_workers * 4))
            chunks = [all_data[i:i+chunk_size] for i in range(0, len(all_data), chunk_size)]
            
            # Process chunks in parallel
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(self._process_chunk_with_meta, chunk, self.image_meta, self.pos_predicate))
            
            # Collect results
            for future in tqdm(futures, desc="Processing chunks"):
                chunk_samples = future.result()
                self.samples.extend(chunk_samples)

        random.shuffle(self.samples)
        
        # Save to cache if enabled
        if cache_dir is not None and use_cache:
            print(f"Saving processed dataset to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f)

    @staticmethod
    def _process_chunk_with_meta(chunk, image_meta, pos_predicate):
        """Static method for processing chunks with image metadata in parallel."""
        chunk_samples = []
        for image_entry in chunk:
            chunk_samples.extend(RelationshipDataset._process_image_static(
                image_entry, image_meta, pos_predicate))
        return chunk_samples

    @staticmethod
    def _process_image_static(image_entry, image_meta, pos_predicate):
        """Static version of _process_image for parallel processing."""
        image_id = image_entry["image_id"]
        if image_id not in image_meta:
            return []

        img_meta = image_meta[image_id]
        img_width, img_height = img_meta["width"], img_meta["height"]
        object_features = {}
        positive_pairs = set()
        samples = []

        # Process all relationships in the image
        for rel in image_entry.get("relationships", []):
            subj_feat = RelationshipDataset._extract_features_static(rel["subject"], img_width, img_height)
            if subj_feat is not None and "object_id" in rel["subject"]:
                object_features[rel["subject"]["object_id"]] = subj_feat

            obj_feat = RelationshipDataset._extract_features_static(rel["object"], img_width, img_height)
            if obj_feat is not None and "object_id" in rel["object"]:
                object_features[rel["object"]["object_id"]] = obj_feat

            if subj_feat is not None and obj_feat is not None:
                predicate = rel["predicate"].lower()
                if predicate == pos_predicate:
                    samples.append((subj_feat, obj_feat, torch.tensor(1.0)))
                    if "object_id" in rel["subject"] and "object_id" in rel["object"]:
                        positive_pairs.add(
                            (rel["subject"]["object_id"], rel["object"]["object_id"])
                        )

        # Generate negative samples only if we have enough objects
        if len(object_features) >= 2:
            # Create lookup dictionaries for faster access
            obj_ids = list(object_features.keys())
            obj_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
            obj_features = list(object_features.values())
            
            # Calculate how many negative samples to generate
            num_neg = min(len(positive_pairs), len(obj_ids) * (len(obj_ids) - 1) // 2)
            
            if num_neg > 0:
                # More efficient negative sample generation
                all_pairs = set()
                for i in range(len(obj_ids)):
                    for j in range(len(obj_ids)):
                        if i != j:  # Exclude self-pairs
                            all_pairs.add((obj_ids[i], obj_ids[j]))
                
                # Filter out positive pairs
                negative_pairs = all_pairs - positive_pairs
                
                # Sample from negative pairs
                negative_pairs = list(negative_pairs)
                if len(negative_pairs) > num_neg:
                    selected_indices = np.random.choice(len(negative_pairs), num_neg, replace=False)
                    selected = [negative_pairs[i] for i in selected_indices]
                else:
                    selected = negative_pairs
                
                # Create negative samples
                for s_id, o_id in selected:
                    s_idx = obj_id_to_idx[s_id]  # O(1) lookup instead of list.index()
                    o_idx = obj_id_to_idx[o_id]
                    samples.append(
                        (obj_features[s_idx], obj_features[o_idx], torch.tensor(0.0))
                    )
        
        return samples

    @staticmethod
    def _extract_features_static(obj, img_width, img_height):
        """Static version of _extract_features for parallel processing."""
        try:
            x = obj["x"] / img_width
            y = obj["y"] / img_height
            w = obj["w"] / img_width
            h = obj["h"] / img_height
            if not (0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0):
                return None
            class_id = float(obj.get("object_id", 0)) % 1000
            return torch.tensor([x, y, w, h, class_id], dtype=torch.float32)
        except (KeyError, TypeError, ValueError):
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj_feat, obj_feat, label = self.samples[idx]
        return subj_feat, obj_feat, label