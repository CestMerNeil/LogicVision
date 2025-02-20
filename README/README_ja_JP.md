[![English](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-usa2x.png)](/README.md)
[![Français](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-france2x.png)](/README/README_fr_FR.md)
[![中文](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-china2x.png)](/README/README_zh_CN.md)
[![日本語](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-japan2x.png)](/README/README_ja_JP.md)

# ロジスティック・テンソル・ニューラル・ネットワークを用いた視覚シーンの理解 🚀 🤖 

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)](https://www.python.org)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-red?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)
[![LTNTorch](https://img.shields.io/badge/Project-LTNTorch-9cf?style=flat-square)](https://github.com/tommasocarraro/LTNtorch)
[![Visual Genome](https://img.shields.io/badge/Data-Visual%20Genome-yellow?style=flat-square)](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)
[![YOLO](https://img.shields.io/badge/Detection-YOLO-orange?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![OneFormer](https://img.shields.io/badge/Segmentation-OneFormer-brightgreen?style=flat-square)](https://github.com/SHI-Labs/OneFormer)

このプロジェクトでは、セグメンテーションモデルとロジックテンソルネットワークを組み合わせ、一階論理式と多層パーセプトロンネットワークにより、画像中のオブジェクトの関係に関する推論を実現し、画像コンテンツ解析の向上を目指します。✨

---

## テクニカルアーキテクチャ

1. **セグメンテーションと特徴抽出**：YOLO フォーム [UltraLytics](https://docs.ultralytics.com) と OneFormer フォーム [SHI-Labs](https://www.shi-labs.com) を使用
2. **対象物のスクリーニング**：関心のある対象物の保持
3. **論理テンソルの生成**：オブジェクトのペアのデカルト積で論理テンソルを生成する
4. **論理的推論**：論理テンソルを使った関係述語推論
5. **結果出力**：推論結果の出力


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

predicate = ["in", "on", "next to"]
for pred in predicate:
    print(f"🚂 {pred} 述語を訓練中 ...")
    trainer(
        pos_predicate=pred,
        neg_predicates=[p for p in predicate if p != pred],
        epoches=50,
        batch_size=32,
        lr=1e-4
    )
```

### 推論の例
```Python
from utils.Inferencer import Inferencer

# 推論器を初期化
analyzer = Inferencer(
    subj_class="person",
    obj_class="bicycle",
    predicate="near"
)

# 画像の単一推論を実行
result = analyzer.inference_single("demo.jpg")
print(f"🔎 Get ：{result['relation']} (Confidence：{result['confidence']:.2f})")

# 画像フォルダの推論を実行
analyzer.process_folder("input_images/")
```

# データベース
[Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)データベースの人間関係と画像メタデータデータを用いて、画像情報と特徴ペア情報を抽出した。

![Visual Genole の例](/README/images/Visual_Genome.png)

このプロジェクトでは、リレーショナル データからデータとターゲットの場所を抽出し、イメージ データを抽出してターゲットの場所を正規化します。

# コード スタイルとドキュメント
このプロジェクトでは、'```black``` と ```isort``` を使用して、一貫したコード スタイルを自動的に適用します。すべてのコード コメントとドキュメントは、[Google Python スタイル ガイド](https://google.github.io/styleguide/) に従って、明瞭性と一貫性を維持します。

送信前にコードを同じ形式に保つには、次のコマンドを使用します。
```bash
black . && isort .
```
# 謝辞
このプロジェクトは [LTNTorch](https://github.com/tommasocarraro/LTNtorch) プロジェクトに基づいており、データ抽出に [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api_beginners_tutorial.html) データベースを使用します。このプロジェクトでは、オブジェクトの検出とセグメンテーションに [YOLO](https://doc.ultralytics.com) および [OneFormer](https://www.shi-labs.com) モデルを使用します。

# ライセンス
このプロジェクトは GNU3.0 ライセンスの下でライセンスされています。詳細については、[LICENSE](/LICENSE) ファイルを参照してください。
---