[![English](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-usa2x.png)](/README.md)
[![Français](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-france2x.png)](/README/README_fr_FR.md)
[![中文](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-china2x.png)](/README/README_zh_CN.md)
[![日本語](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-japan2x.png)](/README/README_ja_JP.md)

# Understanding visual scenes using logistic tensor neural networks 🚀🤖

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)](https://www.python.org)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-red?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)
[![LTNTorch](https://img.shields.io/badge/Project-LTNTorch-9cf?style=flat-square)](https://github.com/tommasocarraro/LTNtorch)
[![Visual Genome](https://img.shields.io/badge/Data-Visual%20Genome-yellow?style=flat-square)](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)
[![YOLO](https://img.shields.io/badge/Detection-YOLO-orange?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![OneFormer](https://img.shields.io/badge/Segmentation-OneFormer-brightgreen?style=flat-square)](https://github.com/SHI-Labs/OneFormer)

This project combines segmentation model and logic tensor network to realize the reasoning of object relationship in images and improve image content analysis through first-order logic formula and multi-layer perceptron network. ✨

---

## Overall architecture and module division
! [Overall Architecture](/README/images/Architecture.png)

1. **✨ Image segmentation and feature extraction**: The YOLO-Seg model from [UltraLytics](https://docs.ultralytics.com) or the OneFormer model from [SHI-Labs](https://www.shi-labs.com) is used to segment and extract features from the input image. image for segmentation and feature extraction.
2. **✨Goal relation detection**: using a logic tensor network from [LTNTorch](https://github.com/tommasocarraro/LTNtorch), each goal is converted into a logical predicate, which is then reasoned over by the logic tensor network.
3. **✨Logical Relationship Training**: Logistic tensor networks were trained using relational data from the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) database.
4. **✨ Output of reasoning results**: reads the relations found by the user using the form of a ternary and outputs the results of the reasoning.


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

predicate = ["in", "on", "next to"]
for pred in predicate:
    print(f"🚂 Training {pred} ...")
    trainer(
        pos_predicate=pred,
        neg_predicates=[p for p in predicate if p != pred],
        epoches=50,
        batch_size=32,
        lr=1e-4
    )
```

### Examples of inference
```Python
from utils.Inferencer import Inferencer

# Initialize the inferencer
analyzer = Inferencer(
    subj_class="person",
    obj_class="bicycle",
    predicate="near"
)

# Perform inference on a single image
result = analyzer.inference_single("demo.jpg")
print(f"🔎 Get ：{result['relation']} (Confidence：{result['confidence']:.2f})")

# Perform inference on a folder of images
analyzer.process_folder("input_images/")
```

# Database
The relationships and image metadata data from the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) database were used to extract image information and feature pair information.

![Visual Genole Example](/README/images/Visual_Genome.png)


The project extracts data and target locations from relational data, and extracts image data to normalize the target locations.

# Code Style and Documentation
This project uses the ```black``` and ```isort``` to automatically enforce a consistent code style. All code comments and documentation follow the [Google Python Style Guide](https://google.github.io/styleguide/) to maintain clarity and consistency.


Use the following command to keep the code in the same format before submitting.
```bash
black . && isort . 
```
# Acknowledgements
This project is based on the [LTNTorch](https://github.com/tommasocarraro/LTNtorch) project and uses the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api_beginners_tutorial.html) database for data extraction. The project uses the [YOLO](https://doc.ultralytics.com) and [OneFormer](https://www.shi-labs.com) models for object detection and segmentation.

# License
This project is licensed under the GNU3.0 License - see the [LICENSE](/LICENSE) file for details.
---
