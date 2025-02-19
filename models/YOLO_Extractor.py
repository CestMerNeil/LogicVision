import pprint
import tomllib
from typing import Dict, List, Union

import torch
from ultralytics import YOLO

with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)


class YOLO_Extractor:
    """YOLO Extractor for object detection.

    This class utilizes a YOLO model to perform object detection on input images.
    It processes the raw model outputs to produce normalized bounding boxes and
    other detection metrics.
    """

    def __init__(self):
        """Initializes the YOLO extractor.

        Loads the model configuration, sets the confidence threshold, and
        initializes the YOLO model along with class labels.
        """
        self.conf_threshold = config["YOLO_Extractor"]["conf_threshold"]
        self.model = YOLO(config["YOLO_Extractor"]["model_path"])
        self.labels = self.model.names
        self.num_classes = len(self.labels)

    def extractor_summary(self):
        """Prints a summary of the YOLO extractor.

        Displays the model details, confidence threshold, class names, and the number
        of classes.
        """
        print(f"Model Path: {self.model}")
        print(f"Confidence Threshold: {self.conf_threshold}")
        print(f"Class Name: {self.labels}")
        print(f"Number of Classes: {self.num_classes}")

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Performs object detection on an input image.

        Args:
            image (torch.Tensor): An image tensor of shape (3, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing processed detection results,
            including normalized bounding boxes, centers, widths, heights, scores, classes,
            masks, the number of detected objects, and the image size.
        """
        result = self.model.predict(image, conf=self.conf_threshold)
        return self._process_result(result[0])

    def _process_result(
        self, result: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Processes raw model output to extract detection results.

        Args:
            result (Dict[str, torch.Tensor]): The raw output from the YOLO model.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing processed detection results with
            keys 'boxes', 'centers', 'widths', 'heights', 'scores', 'classes', 'masks',
            'num_objects', and 'image_size'.
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

            if hasattr(result, "masks") and result.masks is not None:
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
            "boxes": norm_boxes,
            "centers": centers,
            "widths": widths,
            "heights": heights,
            "scores": scores,
            "classes": classes,
            "masks": masks,
            "num_objects": torch.tensor(len(boxes)),
            "image_size": torch.tensor([H, W]),
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
