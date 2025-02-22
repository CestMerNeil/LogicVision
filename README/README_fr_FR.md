[![English](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-usa2x.png)](/README.md)
[![Français](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-france2x.png)](/README/README_fr_FR.md)
[![中文](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-china2x.png)](/README/README_zh_CN.md)
[![日本語](https://cdn3.iconfinder.com/data/icons/142-mini-country-flags-16x16px/32/flag-japan2x.png)](/README/README_ja_JP.md)

# Comprendre les scènes visuelles à l'aide de réseaux de neurones à tenseur logistique 🚀🤖

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)](https://www.python.org)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-red?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)
[![LTNTorch](https://img.shields.io/badge/Project-LTNTorch-9cf?style=flat-square)](https://github.com/tommasocarraro/LTNtorch)
[![Visual Genome](https://img.shields.io/badge/Data-Visual%20Genome-yellow?style=flat-square)](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)
[![YOLO](https://img.shields.io/badge/Detection-YOLO-orange?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![OneFormer](https://img.shields.io/badge/Segmentation-OneFormer-brightgreen?style=flat-square)](https://github.com/SHI-Labs/OneFormer)

Ce projet combine un modèle de segmentation et un réseau tensoriel logique pour réaliser le raisonnement de la relation entre les objets dans les images et améliorer l'analyse du contenu de l'image grâce à une formule logique de premier ordre et à un réseau perceptron multicouche. par le biais d'une formule logique de premier ordre et d'un réseau perceptron multicouche. ✨

---

## Architecture générale et répartition des modules
![Architecture générale](/README/images/Architecture.png)

1) **✨ Segmentation de l'image et extraction des caractéristiques** : Le modèle YOLO-Seg de [UltraLytics](https://docs.ultralytics.com) ou le modèle OneFormer de [SHI-Labs](https://www.shi-labs.com) est utilisé pour la segmentation de l'image d'entrée et l'extraction des caractéristiques. L'image d'entrée est utilisée pour la segmentation et l'extraction des caractéristiques.
2. **✨Goal relation detection** : En utilisant un réseau tensoriel logique de [LTNTorch](https://github.com/tommasocarraro/LTNtorch), chaque objectif est converti en un prédicat logique, qui est ensuite raisonné par le réseau tensoriel logique.
3) **✨Logical Relationship Training** : Les réseaux de tenseurs logistiques ont été entraînés à l'aide de données relationnelles provenant de la base de données [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html).
4. **✨ Sortie des résultats du raisonnement** : lit les relations trouvées par l'utilisateur en utilisant la forme d'un ternaire et sort les résultats du raisonnement.


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

predicate = ["in", "on", "next to"]
for pred in predicate:
    print(f"🚂 Entraîner {pred} ...")
    trainer(
        pos_predicate=pred,
        neg_predicates=[p for p in predicate if p != pred],
        epoches=50,
        batch_size=32,
        lr=1e-4
    )
```

### Exemples de raisonnement
```Python
from utils.Inferencer import Inferencer

# Initialiser l'inférenteur
analyzer = Inferencer(
    subj_class="person",
    obj_class="bicycle",
    predicate="near"
)

# Effectuer l'inférence sur une seule image
result = analyzer.inference_single("demo.jpg")
print(f"🔎 Obtenu ：{result['relation']} (Confiance：{result['confidence']:.2f})")

# Effectuer une inférence sur un dossier d'images
analyzer.process_folder("input_images/")
```

# Base de données
Les relations et les métadonnées des images de la base de données [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) ont été utilisées pour extraire les informations sur les images et les paires de caractéristiques. ont été utilisées pour extraire des informations sur les images et des informations sur les paires de caractéristiques.

![Visual Genole Example](/README/images/Visual_Genome.png)

Le projet extrait les données et les emplacements cibles à partir de données relationnelles et extrait les données d'image pour normaliser les emplacements cibles.

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