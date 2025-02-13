[![English](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-usa2x.png)](/README.md)
[![Français](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-france2x.png)](/README/README_fr_FR.md)
[![中文](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-china2x.png)](/README/README_zh_CN.md)
[![日本語](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-japan2x.png)](/README/README_ja_JP.md)

# ロジスティック・テンソル・ニューラル・ネットワークを用いた視覚シーンの理解 🚀 🤖 

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)](https://www.python.org)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-red?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)
[![LTNTorch](https://img.shields.io/badge/Project-LTNTorch-9cf?style=flat-square)](https://github.com/ltntorch)
[![Visual Genome](https://img.shields.io/badge/Data-Visual%20Genome-yellow?style=flat-square)](https://visualgenome.org)
[![YOLO](https://img.shields.io/badge/Detection-YOLO-orange?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![OneFormer](https://img.shields.io/badge/Segmentation-OneFormer-brightgreen?style=flat-square)](https://github.com/isl-org/OneFormer)

このプロジェクトでは、セグメンテーションモデルとロジックテンソルネットワークを組み合わせ、一階論理式と多層パーセプトロンネットワークにより、画像中のオブジェクトの関係に関する推論を実現し、画像コンテンツ解析の向上を目指します。✨

---

## テクニカルアーキテクチャ

1.**セグメンテーションと特徴抽出**：YOLOv8 / OneFormerを使用
2.**対象物のスクリーニング**：関心のある対象物の保持
3.**論理テンソルの生成**：オブジェクトのペアのデカルト積で論理テンソルを生成する
4.**論理的推論**：論理テンソルを使った関係述語推論
5.**結果出力**：推論結果の出力


## インストレーションガイド

### トレーニング環境（Ubuntu 22.04）
```bash
pip install -r requirements.train.txt
```

### 推論環境（macOS 15.3）
```bash
pip install -r requirements.inference.txt
```

YOLOとOneFormerの訓練済みモデルは、プログラムの実行時に自動的にダウンロードされます。

## 使用に関するガイドライン

### トレーニングの例
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

### 推論の例
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

# データベース
[Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)データベースの人間関係と画像メタデータデータを用いて、画像情報と特徴ペア情報を抽出した。

---