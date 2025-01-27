# Visual Scene Understanding with Logic Tensor Networks

This project integrates advanced segmentation models with Logic Tensor Networks to enhance image content analysis for researchers. By leveraging the segmentation results and logical reasoning, the system identifies specific scene targets through first-order logic formulas and spatial constraints like Euclidean distance.

The goal is to improve the interpretability and utility of image analysis. This pipeline enables researchers to filter and analyze images with high precision, providing deeper insights into complex visual data.

# Features Extractor
The feature extractor is a pre-trained model that extracts features from images. The features are then used to generate segmentation masks and identify objects in the scene.

In this project, we use [Ultralytics YOLO](https://docs.ultralytics.com) and [OneFormer](https://praeclarumjj3.github.io/oneformer/) as feature extractors. These models are pre-trained on large datasets and can accurately detect objects in images.

