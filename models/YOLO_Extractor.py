import torch
import pprint
import tomllib
from pathlib import Path
from typing import Dict, List, Union

from ultralytics import YOLO

with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)

pprint.pprint(f"Config List: {config}")

class YOLO_Extractor:
    def __init__(self):
        self.conf_threshold = config['YOLO_Extractor']['conf_threshold']
        self.model = YOLO(config['YOLO_Extractor']['model_path'])
        self.class_name = self.model.names
        self.num_classes = len(self.class_name)

    def extractor_summary(self):
        """
        Print the summary of the YOLO Extractor
        """
        print(f"Model Path: {self.model}")
        print(f"Confidence Threshold: {self.conf_threshold}")
        print(f"Class Name: {self.class_name}")
        print(f"Number of Classes: {self.num_classes}")

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict the image using the YOLO model

        Args:
            image (torch.Tensor): Image of shape (3, H, W)
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the results
        """
        result = self.model.predict(image, conf=self.conf_threshold)
        return self._process_result(result[0])

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict the images using the YOLO model and align the results for batch processing.

        Args:
            images (torch.Tensor): Images of shape (N, 3, H, W)
        
        Returns:
            Dict[str, torch.Tensor]: Batch results with aligned tensors
        """
        results = self.model.predict(images, conf=self.conf_threshold)
        
        batch_results = [self._process_result(result) for result in results]
        
        max_objects = max(res['num_objects'].item() for res in batch_results)
        batch_size = len(batch_results)
        
        padded_boxes = torch.zeros((batch_size, max_objects, 4))
        padded_centers = torch.zeros((batch_size, max_objects, 2))
        padded_widths = torch.zeros((batch_size, max_objects))
        padded_heights = torch.zeros((batch_size, max_objects))
        padded_scores = torch.zeros((batch_size, max_objects))
        padded_classes = torch.zeros((batch_size, max_objects))
        padded_masks = None

        H, W = images.shape[2:]  # Assume all images in batch have the same size

        for i, res in enumerate(batch_results):
            num_objects = res['num_objects'].item()
            
            if num_objects > 0:
                padded_boxes[i, :num_objects] = res['boxes']
                padded_centers[i, :num_objects] = res['centers']
                padded_widths[i, :num_objects] = res['widths']
                padded_heights[i, :num_objects] = res['heights']
                padded_scores[i, :num_objects] = res['scores']
                padded_classes[i, :num_objects] = res['classes']
                
                if res['masks'] is not None:
                    if padded_masks is None:
                        padded_masks = torch.zeros((batch_size, max_objects, H, W))
                    padded_masks[i, :num_objects] = res['masks']
        
        if padded_masks is None:
            padded_masks = torch.zeros((batch_size, max_objects, H, W))
        
        return {
            'boxes': padded_boxes,
            'centers': padded_centers,
            'widths': padded_widths,
            'heights': padded_heights,
            'scores': padded_scores,
            'classes': padded_classes,
            'masks': padded_masks,
            'num_objects': torch.tensor([res['num_objects'].item() for res in batch_results]),
            'image_size': torch.tensor([H, W]),
        }

    def _process_result(self, result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process the result of the YOLO model

        Args:
            result (Dict[str, torch.Tensor]): Result of the YOLO model
            
        Returns:
            Dict[str, torch.Tensor]: Processed result
        """
        H, W = result.orig_shape[:2]

        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy
            scores = result.boxes.conf
            classes = result.boxes.cls

            boxes = torch.as_tensor(boxes)
            scores = torch.as_tensor(scores)
            classes = torch.as_tensor(classes)

            norm_boxes = boxes.clone()
            norm_boxes[:, [0, 2]] /= W
            norm_boxes[:, [1, 3]] /= H

            centers = (norm_boxes[:, :2] + norm_boxes[:, 2:]) / 2
            widths = norm_boxes[:, 2] - norm_boxes[:, 0]
            heights = norm_boxes[:, 3] - norm_boxes[:, 1]

            if hasattr(result, 'masks') and result.masks is not None:
                masks = torch.as_tensor(result.masks.data)
            else:
                masks = torch.zeros((len(boxes), H, W))
        else:
            boxes = torch.zeros((0, 4))
            norm_boxes = torch.zeros((0, 4))
            centers = torch.zeros((0, 2))
            widths = torch.zeros((0,))
            heights = torch.zeros((0,))
            scores = torch.zeros((0,))
            classes = torch.zeros((0,))
            masks = torch.zeros((0, H, W))

        return {
            'boxes': norm_boxes,
            'centers': centers,
            'widths': widths,
            'heights': heights,
            'scores': scores,
            'classes': classes,
            'masks': masks,
            'num_objects': torch.tensor(len(boxes)),
            'image_size': torch.tensor([H, W])
        }

if __name__ == "__main__":
    import os
    import torch

    extractor = YOLO_Extractor()
    print("Initialized YOLO Extractor.")
    extractor.extractor_summary()

    single_image_path = "images/image2.jpg"
    if not os.path.exists(single_image_path):
        raise FileNotFoundError(f"Image not found: {single_image_path}")

    print(f"\nTesting single image prediction for {single_image_path}...")
    single_image_tensor = torch.rand(3, 640, 640)
    single_output = extractor.predict(single_image_tensor)
    print("Single image prediction result keys:")
    for key, value in single_output.items():
        print(f"  {key}: {value.shape if isinstance(value, torch.Tensor) else value}")

    batch_images = torch.rand(4, 3, 640, 640)
    print("\nTesting batch image prediction...")
    batch_output = extractor.predict_batch(batch_images)
    print("Batch prediction result keys:")
    for key, value in batch_output.items():
        print(f"  {key}: {value.shape if isinstance(value, torch.Tensor) else value}")