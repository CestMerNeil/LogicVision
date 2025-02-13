[![English](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-usa2x.png)](/README.md)
[![Français](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-france2x.png)](/README/README_fr_FR.md)
[![中文](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-china2x.png)](/README/README_zh_CN.md)
[![日本語](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-japan2x.png)](/README/README_ja_JP.md)

# Comprendre les scènes visuelles à l'aide de réseaux de neurones à tenseur logistique 🚀🤖🤖

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)](https://www.python.org)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-red?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)
[![LTNTorch](https://img.shields.io/badge/Project-LTNTorch-9cf?style=flat-square)](https://github.com/ltntorch)
[![Visual Genome](https://img.shields.io/badge/Data-Visual%20Genome-yellow?style=flat-square)](https://visualgenome.org)
[![YOLO](https://img.shields.io/badge/Detection-YOLO-orange?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![OneFormer](https://img.shields.io/badge/Segmentation-OneFormer-brightgreen?style=flat-square)](https://github.com/isl-org/OneFormer)

Ce projet combine un modèle de segmentation et un réseau tensoriel logique pour réaliser le raisonnement de la relation entre les objets dans les images et améliorer l'analyse du contenu de l'image grâce à une formule logique de premier ordre et à un réseau perceptron multicouche. par le biais d'une formule logique de premier ordre et d'un réseau perceptron multicouche. ✨

---

## Architecture technique

1. **Segmentation et extraction de caractéristiques** : utilisation de YOLOv8 / OneFormer
2. **Criblage des objets** : conservation des objets présentant un intérêt
3. **Génération de tenseurs logiques** : produit cartésien de paires d'objets pour générer un tenseur logique
4. **Raisonnement logique** : raisonnement par prédicats relationnels à l'aide du tenseur logique
5) **Sortie des résultats** : sortie des résultats du raisonnement


## Guide d'installation

### Environnement de formation (Ubuntu 22.04)
```bash
pip install -r requirements.train.txt
```

### Environnement de raisonnement (macOS 15.3)
```bash
pip install -r requirements.inference.txt
```

Des modèles pré-entraînés pour YOLO et OneFormer sont automatiquement téléchargés lors de l'exécution du programme.

## Lignes directrices pour l'utilisation

### Exemple de formation
```Python
from utils.Trainer import trainer
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

predicates = ["in", "on", "next to", "on top of", "near", "under"]
params = config["Train"]

for pred in predicates:
    trainer(
        pos_predicate=pred,
        neg_predicates=[p for p in predicates if p != pred],
        epoches=params["epochs"],
        batch_size=params["batch_size"],
        lr=params["lr"]
    )
```

### Exemples de raisonnement
```Python
from PIL import Image
from utils.Inferencer import Inferencer
from utils.Draw import draw_and_save_result

inferencer = Inferencer(subj_class="person", obj_class="sky", predicate="near")
image = Image.open("path_to_image.jpg")
result = inferencer.inference_single(image)

if result.get("exists", True):
    draw_and_save_result(image, result, "result.jpg")
```

# Base de données
Les relations et les métadonnées des images de la base de données [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) ont été utilisées pour extraire les informations sur les images et les paires de caractéristiques. ont été utilisées pour extraire des informations sur les images et des informations sur les paires de caractéristiques.

---