# utils/DataLoader.py
import torch
import tomllib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from PIL import Image
from pathlib import Path
import os
import pandas as pd
 
# Using Open Image Datasets V7 by Fiftyone
# import fiftyone as fo
# import fiftyone.zoo as foz

with open("config.toml", "rb") as f:
    config = tomllib.load(f)
    data_config = config["DataLoader"]


###########################################
# Visual Genome
###########################################
class VisualGenomeDataLoader(Dataset):
    def __init__(self, images_dir, relationships_json, img_size):
        self.img_size = img_size
        self.images_dir = images_dir
        self.relationships = pd.read_json(relationships_json)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        with open(relationships_json, 'r', encoding='utf-8') as f:
            self.relationships = json.load(f)
        self.data_dict = {str(item['image_id']): item for item in self.relationships}
        


###########################################
# Open Image Datasets V7 by Fiftyone
# ATTENTION: The relationship annotations in this dataset are not answered to our needs.
###########################################

# Download Open Image Dataset V7
def download_openimages_v7():
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["relationships"],
        max_samples=5000,
        overwrite=False
    )
    session = fo.launch_app(dataset)

    return dataset

class OpenImagesDataLoader(Dataset):
    def __init__(self, fo_dataset, transform=None):
        self.fo_dataset = fo_dataset
        self.samples = list(fo_dataset)
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)

        relationships = getattr(sample, "relationships", None)
        if relationships is None:
            relationships = []
        return image, relationships


###########################################
# VOC 2012 Dataset by Torchvision
###########################################

class VisionDataset(Dataset):
    def __init__(self, split='train'):
        data_root = Path(data_config["data_root"])
        self.img_dir = data_root / split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"{self.img_dir} not found")

        self.img_paths = [
            p for p in self.img_dir.glob("*") if p.suffix in [".jpg", ".png", ".jpeg"]
        ]
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((data_config["img_size"], data_config["img_size"])),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img

def get_dataloader(split='train'):
    valid_splits = ['train', 'val', 'test']
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}")
    
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    dataset = VisionDataset(split)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=(split == 'train'),
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

if __name__ == "__main__":
    import ijson

    with open('/Volumes/T7_Shield/Datasets/relationships.json', 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'item')  # 修改 'item' 为 JSON 文件中的根元素
        for i, item in enumerate(objects):
            if i >= 1:
                break
            print(item)