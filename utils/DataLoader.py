import torch
import json
from torch.utils.data import Dataset
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class RelationshipDataset(Dataset):
    """
    @brief PyTorch dataset for image relationship data.
    
    This dataset processes relationship data from a JSON file along with image metadata,
    generating positive (and implicitly negative) samples based on the specified predicate.
    
    @param relationships_json_path Path to the JSON file containing relationship data.
    @param image_meta_json_path Path to the JSON file containing image metadata.
    @param pos_predicate The predicate string used to identify positive relationships.
    @param use_cuda Boolean flag to indicate whether to use CUDA if available.
    """
    def __init__(self, relationships_json_path, image_meta_json_path, pos_predicate, use_cuda=True):
        self.samples = []
        self.pos_predicate = pos_predicate.lower()
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        def chunked_json_loader(path, chunk_size=1000):
            """
            @brief Generator function to load JSON data in chunks.
            
            @param path Path to the JSON file.
            @param chunk_size Number of items per chunk.
            @return Generator yielding chunks of JSON data.
            """
            with open(path, 'r') as f:
                data = json.load(f)
                for i in range(0, len(data), chunk_size):
                    yield data[i:i + chunk_size]

        # Load image metadata and create a mapping from image_id to metadata.
        with open(image_meta_json_path, 'r') as f:
            self.image_meta = {img["image_id"]: img for img in json.load(f)}

        # Process relationship data in parallel.
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for chunk in chunked_json_loader(relationships_json_path):
                futures.append(executor.submit(self.process_chunk, chunk))
            
            for future in tqdm(futures, desc="Processing chunks"):
                chunk_samples = future.result()
                self.samples.extend(chunk_samples)

        # Shuffle samples globally.
        random.shuffle(self.samples)

    def process_chunk(self, chunk):
        """
        @brief Process a chunk of image relationship entries.
        
        @param chunk A list of image relationship entries.
        @return List of processed samples from the chunk.
        """
        chunk_samples = []
        for image_entry in chunk:
            chunk_samples.extend(self._process_image(image_entry))
        return chunk_samples

    def _process_image(self, image_entry):
        """
        @brief Process relationship data for a single image.
        
        @param image_entry Dictionary containing image relationship data.
        @return List of samples (subject feature, object feature, label) for the image.
        """
        image_id = image_entry["image_id"]
        if image_id not in self.image_meta:
            return []
        
        img_meta = self.image_meta[image_id]
        img_width, img_height = img_meta["width"], img_meta["height"]
        object_features = {}
        positive_pairs = set()
        samples = []

        # Extract features for each relationship.
        for rel in image_entry.get("relationships", []):
            subj_feat = self._extract_features(rel["subject"], img_width, img_height)
            if subj_feat is not None and "object_id" in rel["subject"]:
                object_features[rel["subject"]["object_id"]] = subj_feat
            
            obj_feat = self._extract_features(rel["object"], img_width, img_height)
            if obj_feat is not None and "object_id" in rel["object"]:
                object_features[rel["object"]["object_id"]] = obj_feat

            if subj_feat is not None and obj_feat is not None:
                predicate = rel["predicate"].lower()
                if predicate == self.pos_predicate:
                    samples.append((subj_feat, obj_feat, torch.tensor(1.0)))
                    if "object_id" in rel["subject"] and "object_id" in rel["object"]:
                        positive_pairs.add((rel["subject"]["object_id"], rel["object"]["object_id"]))

        # Generate negative samples if possible.
        if len(object_features) >= 2:
            obj_ids = list(object_features.keys())
            obj_features = list(object_features.values())
            num_neg = min(len(positive_pairs), len(obj_ids) * (len(obj_ids) - 1) // 2)
            if num_neg > 0:
                indices = np.array(np.meshgrid(np.arange(len(obj_ids)), np.arange(len(obj_ids)))).T.reshape(-1, 2)
                mask = (indices[:, 0] != indices[:, 1])
                candidate_pairs = indices[mask]

                candidate_ids = [(obj_ids[i], obj_ids[j]) for i, j in candidate_pairs]
                valid_mask = [pair not in positive_pairs for pair in candidate_ids]
                valid_pairs = np.array(candidate_ids)[valid_mask]

                if len(valid_pairs) > num_neg:
                    selected = valid_pairs[np.random.choice(len(valid_pairs), num_neg, replace=False)]
                else:
                    selected = valid_pairs

                for s_id, o_id in selected:
                    s_idx = obj_ids.index(s_id)
                    o_idx = obj_ids.index(o_id)
                    samples.append((obj_features[s_idx], obj_features[o_idx], torch.tensor(0.0)))

        return samples

    def _extract_features(self, obj, img_width, img_height):
        """
        @brief Extract normalized features from an object.
        
        The features include normalized x, y coordinates, width, height, and a limited class identifier.
        
        @param obj Dictionary representing an object with bounding box information.
        @param img_width Width of the image.
        @param img_height Height of the image.
        @return Torch tensor containing extracted features, or None if the object is invalid.
        """
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
        """
        @brief Return the number of samples in the dataset.
        
        @return Total number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        @brief Retrieve a sample by index.
        
        @param idx Index of the sample.
        @return Tuple (subject feature, object feature, label).
        """
        subj_feat, obj_feat, label = self.samples[idx]
        return subj_feat, obj_feat, label