# utils/DataLoader.py
import torch
import tomllib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from PIL import Image
from pathlib import Path
import os
import pandas as pd

with open("config.toml", "rb") as f:
    config = tomllib.load(f)
    data_config = config["DataLoader"]

class VisionGenomeDataset(Dataset):
    def __init__(self, path_json, path_img, transform=None):
        self.path_json = path_json
        self.path_img = path_img
        self.transform = transform

        
