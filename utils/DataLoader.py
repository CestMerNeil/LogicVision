# utils/DataLoader.py
import torch
import tomllib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from PIL import Image
from pathlib import Path
import os

with open("config.toml", "rb") as f:
    config = tomllib.load(f)
    data_config = config["DataLoader"]

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
