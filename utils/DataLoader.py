import json
import random
import os
import pickle
import gc
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import ijson  # Add this import for streaming JSON parsing

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
        max_samples=None,
        max_neg_per_image=100,
    ):
        """Initializes the RelationshipDataset and processes the data with optimized performance.
        
        Args:
            relationships_json_path: Path to the relationships JSON file
            image_meta_json_path: Path to the image metadata JSON file
            pos_predicate: The positive predicate to use for relationship classification
            cache_dir: Directory to store cached dataset (default: None)
            use_cache: Whether to use cached dataset if available (default: True)
            num_workers: Number of worker processes for parallel processing (default: None)
            max_samples: Maximum number of samples to include in the dataset (default: None)
            max_neg_per_image: Maximum number of negative samples per image (default: 100)
        """
        self.samples = []
        self.pos_predicate = pos_predicate.lower()
        self.max_neg_per_image = max_neg_per_image
        
        # Create cache filename based on inputs
        if cache_dir is not None and use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = f"{os.path.basename(relationships_json_path)}_{os.path.basename(image_meta_json_path)}_{pos_predicate.replace(' ', '_')}"
            if max_samples:
                cache_name += f"_max{max_samples}"
            cache_path = os.path.join(cache_dir, f"{cache_name}.pkl")
            
            # Try loading from cache first
            if os.path.exists(cache_path):
                print(f"Loading cached dataset from {cache_path}")
                try:
                    with open(cache_path, 'rb') as f:
                        self.samples = pickle.load(f)
                        random.shuffle(self.samples)
                        print(f"Loaded {len(self.samples)} samples from cache")
                        return
                except (pickle.UnpicklingError, EOFError):
                    print("Cache file corrupt, regenerating dataset...")

        # If not using cache or cache doesn't exist, process data
        print("Processing dataset from source files...")
        
        # Load image metadata once
        print(f"Loading image metadata from {image_meta_json_path}")
        with open(image_meta_json_path, "r") as f:
            self.image_meta = {img["image_id"]: img for img in json.load(f)}
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        num_workers = min(num_workers or os.cpu_count(), 8)  # Limit to reasonable number
        
        # Process in batches to reduce memory usage
        self._process_relationships_in_batches(
            relationships_json_path, 
            num_workers, 
            max_samples
        )
        
        # Clean up to free memory
        gc.collect()
        
        print(f"Final dataset contains {len(self.samples)} samples")
        random.shuffle(self.samples)
        
        # Save to cache if enabled
        if cache_dir is not None and use_cache:
            print(f"Saving processed dataset to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f)
    
    def _process_relationships_in_batches(self, relationships_json_path, num_workers, max_samples):
        """Process relationships file in batches to reduce memory usage."""
        total_processed = 0
        batch_size = 1000  # Process 1000 images at a time
        
        try:
            # First try to get total count for progress bar
            with open(relationships_json_path, 'r') as f:
                # Count items in JSON array without loading entire file
                total_images = sum(1 for _ in ijson.items(f, 'item'))
            
            with open(relationships_json_path, 'r') as f:
                # Stream-process the file
                batch = []
                
                for image_entry in tqdm(ijson.items(f, 'item'), total=total_images, desc="Processing images"):
                    batch.append(image_entry)
                    
                    if len(batch) >= batch_size:
                        self._process_batch(batch, num_workers)
                        total_processed += len(batch)
                        batch = []
                        
                        # Check if we've reached max_samples
                        if max_samples and len(self.samples) >= max_samples:
                            break
                
                # Process any remaining items
                if batch and (not max_samples or len(self.samples) < max_samples):
                    self._process_batch(batch, num_workers)
        
        except (ijson.JSONError, json.JSONDecodeError) as e:
            # Fall back to regular JSON loading if streaming fails
            print(f"Streaming JSON parsing failed: {e}")
            print("Falling back to regular JSON loading (higher memory usage)")
            
            with open(relationships_json_path, "r") as f:
                all_data = json.load(f)
            
            # Process in multiple batches
            for i in range(0, len(all_data), batch_size):
                batch = all_data[i:i+batch_size]
                self._process_batch(batch, num_workers)
                
                # Check if we've reached max_samples
                if max_samples and len(self.samples) >= max_samples:
                    break
        
        # Trim samples if needed
        if max_samples and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)

    def _process_batch(self, batch, num_workers):
        """Process a batch of images using multiple workers."""
        if not batch:
            return
            
        # For very small batches, just process directly
        if len(batch) <= 10 or num_workers <= 1:
            for image_entry in batch:
                samples = self._process_image_static(
                    image_entry, 
                    self.image_meta, 
                    self.pos_predicate,
                    self.max_neg_per_image
                )
                self.samples.extend(samples)
            return
            
        # For larger batches, use parallel processing
        chunk_size = max(1, len(batch) // (num_workers * 2))
        chunks = [batch[i:i+chunk_size] for i in range(0, len(batch), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(
                    self._process_chunk_with_meta, 
                    chunk, 
                    self.image_meta, 
                    self.pos_predicate,
                    self.max_neg_per_image
                ))
            
            for future in futures:
                chunk_samples = future.result()
                self.samples.extend(chunk_samples)

    @staticmethod
    def _process_chunk_with_meta(chunk, image_meta, pos_predicate, max_neg_per_image):
        """Static method for processing chunks with image metadata in parallel."""
        chunk_samples = []
        for image_entry in chunk:
            chunk_samples.extend(RelationshipDataset._process_image_static(
                image_entry, image_meta, pos_predicate, max_neg_per_image))
        return chunk_samples

    @staticmethod
    def _process_image_static(image_entry, image_meta, pos_predicate, max_neg_per_image=100):
        """Static version of _process_image for parallel processing."""
        image_id = image_entry.get("image_id")
        if not image_id or image_id not in image_meta:
            return []

        img_meta = image_meta[image_id]
        img_width, img_height = img_meta.get("width", 0), img_meta.get("height", 0)
        if not img_width or not img_height:
            return []
            
        object_features = {}
        positive_pairs = set()
        samples = []

        # Process all relationships in the image
        for rel in image_entry.get("relationships", []):
            # Skip invalid relationships
            if not isinstance(rel, dict) or "subject" not in rel or "object" not in rel or "predicate" not in rel:
                continue
                
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
            
            # Calculate how many negative samples to generate (with stricter limits)
            num_neg = min(
                len(positive_pairs),
                len(obj_ids) * (len(obj_ids) - 1) // 4,  # Reduced from //2 to //4
                max_neg_per_image
            )
            
            if num_neg > 0:
                # More memory-efficient approach: randomly sample pairs instead of creating all pairs
                negative_pairs = []
                existing_pairs = set(positive_pairs)
                attempts = 0
                max_attempts = num_neg * 5  # Limit attempts to avoid infinite loops
                
                while len(negative_pairs) < num_neg and attempts < max_attempts:
                    attempts += 1
                    i = random.randrange(len(obj_ids))
                    j = random.randrange(len(obj_ids))
                    if i != j:
                        pair = (obj_ids[i], obj_ids[j])
                        if pair not in existing_pairs and pair not in negative_pairs:
                            negative_pairs.append(pair)
                            # Create negative samples directly to avoid storing large lists
                            s_idx = obj_id_to_idx[pair[0]]
                            o_idx = obj_id_to_idx[pair[1]]
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