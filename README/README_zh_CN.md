[![English](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-usa2x.png)](/README.md)
[![Français](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-france2x.png)](/README/README_fr_FR.md)
[![中文](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-china2x.png)](/README/README_zh_CN.md)
[![日本語](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-japan2x.png)](/README/README_ja_JP.md)

# 利用逻辑张量神经网络理解视觉场景 🚀🤖

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)](https://www.python.org)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-red?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)
[![LTNTorch](https://img.shields.io/badge/Project-LTNTorch-9cf?style=flat-square)](https://github.com/tommasocarraro/LTNtorch)
[![Visual Genome](https://img.shields.io/badge/Data-Visual%20Genome-yellow?style=flat-square)](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)
[![YOLO](https://img.shields.io/badge/Detection-YOLO-orange?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![OneFormer](https://img.shields.io/badge/Segmentation-OneFormer-brightgreen?style=flat-square)](https://github.com/SHI-Labs/OneFormer)

本项目结合分割模型与逻辑张量网络，通过一阶逻辑公式和多层感知机网络，实现图像中物体关系的推理，提升图像内容分析能力。✨

---

## 技术架构

1. **分割与特征提取**：使用来自 [UltraLytics](https://docs.ultralytics.com) 的 YOLO 和来自 [SHI-Labs](https://www.shi-labs.com) 的 OneFormer
2. **物体筛选**：保留感兴趣的物体  
3. **逻辑张量生成**：对物体对进行笛卡尔积生成逻辑张量  
4. **逻辑推理**：利用逻辑张量进行关系谓词推理  
5. **结果输出**：输出推理结果


## 安装指南

### 训练环境 (Ubuntu 22.04)
```bash
pip install -r requirements.train.txt
```

### 推理环境 (macOS 15.3)
```bash
pip install -r requirements.inference.txt
```

程序运行时会自动下载YOLO和OneFormer的预训练模型。

## 使用指南

### 训练示例
```Python
from utils.Trainer import trainer

predicate = ["in", "on", "next to"]
for pred in predicate:
    print(f"🚂 正在训练 {pred} ...")
    trainer(
        pos_predicate=pred,
        neg_predicates=[p for p in predicate if p != pred],
        epoches=50,
        batch_size=32,
        lr=1e-4
    )
```

### 推理示例
```Python
from utils.Inferencer import Inferencer

# 初始化推理器
analyzer = Inferencer(
    subj_class="person",
    obj_class="bicycle",
    predicate="near"
)

# 对单张图片进行推理
result = analyzer.inference_single("demo.jpg")
print(f"🔎 存在 ：{result['relation']} (置信度：{result['confidence']:.2f})")

# 对图片文件夹进行推理
analyzer.process_folder("input_images/")
```

# 数据库
使用 [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) 数据库中的 relationships 和 image metadata 数据来提取图像信息及特征对信息。

![Visual Genole 示例](/README/images/Visual_Genome.png)

该项目从关系数据中提取数据和目标位置，并提取图像数据以规范化目标位置。

# 代码风格和文档
项目使用 ```black```和 ```isort``` 自动强制执行一致的代码风格。所有代码注释和文档均遵循 [Google Python 风格指南](https://google.github.io/styleguide/) 以保持清晰度和一致性。

在提交之前，使用以下命令使代码保持相同的格式。
```bash
black . && isort .
```
# 致谢
该项目基于 [LTNTorch](https://github.com/tommasocarraro/LTNtorch) 项目，并使用 [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api_beginners_tutorial.html) 数据库进行数据提取。该项目使用 [YOLO](https://doc.ultralytics.com) 和 [OneFormer](https://www.shi-labs.com) 模型进行对象检测和分割。

# 许可证
该项目根据 GNU3.0 许可证获得许可 - 有关详细信息，请参阅 [LICENSE](LICENSE) 文件。
---

