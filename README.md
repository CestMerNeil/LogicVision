[![English](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-usa2x.png)](/README.md)
[![Français](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-france2x.png)](/README/README_fr_FR.md)
[![中文](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-china2x.png)](/README/README_zh_CN.md)
[![日本語](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-japan2x.png)](/README/README_ja_JP.md)

# Understanding visual scenes using logistic tensor neural networks 🚀🤖

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)](https://www.python.org)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-red?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)
[![LTNTorch](https://img.shields.io/badge/Project-LTNTorch-9cf?style=flat-square)](https://github.com/ltntorch)
[![Visual Genome](https://img.shields.io/badge/Data-Visual%20Genome-yellow?style=flat-square)](https://visualgenome.org)
[![YOLO](https://img.shields.io/badge/Detection-YOLO-orange?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![OneFormer](https://img.shields.io/badge/Segmentation-OneFormer-brightgreen?style=flat-square)](https://github.com/isl-org/OneFormer)

This project combines segmentation model and logic tensor network to realize the reasoning of object relationship in images and improve image content analysis through first-order logic formula and multi-layer perceptron network. ✨

---

## Technical architecture

1. **Segmentation and feature extraction**: using YOLOv8 / OneFormer
2. **Screening of objects**: retention of objects of interest
3. **Logic tensor generation**: Cartesian product of pairs of objects to generate a logic tensor
4. **Logical reasoning**: relational predicate reasoning using the logic tensor
5. **Output of results**: output of reasoning results


## Installation Guide

### Training environment (Ubuntu 22.04)
```bash
pip install -r requirements.train.txt
```

### Reasoning environment (macOS 15.3)
```bash
pip install -r requirements.inference.txt
```

Pre-trained models for YOLO and OneFormer are automatically downloaded when the program is run.

## Guidelines for use

### Example of training
```Python
from utils.Trainer import trainer
import tomllib

with open("config.toml", "rb") as f.
    config = tomllib.load(f)

predicates = ["in", "on", "next to", "on top of", "near", "under"]
params = config["Train"]

for pred in predicates:
    trainer(
        pos_predicate=pred,
        neg_predicates=[p for p in predicates if p ! = pred],
        epoches=params["epochs"],
        batch_size=params["batch_size"],
        lr=params["lr"]
    )
```

### Examples of reasoning
```Python
from PIL import Image
from utils.Inferencer import Inferencer
from utils.Draw import draw_and_save_result

inferencer = Inferencer(subj_class="person", obj_class="sky", predicate="near")
image = Image.open("path_to_image.jpg")
result = inferencer.inference_single(image)

if result.get("exists", True): if result.get("exists", True).
    draw_and_save_result(image, result, "result.jpg")
```

# Database
The relationships and image metadata data from the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) database were used to extract image information and feature pair information.

---
