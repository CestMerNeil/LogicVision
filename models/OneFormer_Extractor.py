import pprint
import tomllib
from typing import Dict, List

import torch
from PIL import Image
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)


class OneFormer_Extractor:
    """Extractor for universal segmentation using the OneFormer model.

    This class initializes the OneFormer processor and model using pretrained weights
    specified in the configuration file. It provides methods to perform prediction on
    a single image and process the model outputs into instance-level segmentation results.
    """

    def __init__(self):
        """Initialize the OneFormer extractor.

        Loads the processor and model from pretrained weights as defined in the configuration,
        and sets up thresholds and label information.
        """
        self.processor = OneFormerProcessor.from_pretrained(
            config["OneFormer_Extractor"]["processor"]
        )
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            config["OneFormer_Extractor"]["model"]
        )
        self.conf_threshold = config["OneFormer_Extractor"]["conf_threshold"]
        self.mask_threshold = config["OneFormer_Extractor"]["mask_threshold"]
        self.labels = self.model.config.id2label
        self.num_classes = len(self.labels)

    def extractor_summary(self):
        """Print a summary of the OneFormer extractor components.

        Displays information about the processor, model, labels, and number of classes.
        """
        print(f"Processor: {self.processor}")
        print(f"Model: {self.model}")
        print(f"Labels: {self.labels}")
        print(f"Number of Classes: {self.num_classes}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Predict segmentation instances in a given image.

        Processes an input image using the OneFormer processor and model, and returns the
        instance-level detection outputs including bounding boxes, centers, sizes, scores,
        and class information.

        Args:
            image (Image.Image): The input image to process.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing instance-level segmentation results.
        """
        inputs = self.processor(
            images=image,
            task_inputs=[config["OneFormer_Extractor"]["task_inputs"]],
            return_tensors="pt",
        )
        outputs = self.model(**inputs)
        return self._process_result(outputs, image_size=image.size[::-1])

    def _process_result(self, outputs, image_size: tuple) -> Dict[str, torch.Tensor]:
        """Process raw model outputs to generate instance-level segmentation results.

        Args:
            outputs: Raw outputs from the OneFormer model.
            image_size (tuple): The target image size as (height, width).

        Returns:
            Dict[str, torch.Tensor]: Processed segmentation results with instance-level details.
        """
        H, W = image_size
        processed = self.processor.post_process_panoptic_segmentation(
            outputs,
            threshold=self.conf_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[(H, W)],
        )[0]

        panoptic_seg = processed["segmentation"]
        segments_info = processed["segments_info"]

        instance_res = self._panoptic_to_instance(panoptic_seg, segments_info)
        return instance_res

    def _panoptic_to_instance(
        self, panoptic_seg: torch.Tensor, segments_info: List[dict]
    ) -> Dict[str, torch.Tensor]:
        """Convert panoptic segmentation results to instance-level outputs.

        Args:
            panoptic_seg (torch.Tensor): The panoptic segmentation tensor.
            segments_info (List[dict]): A list of dictionaries with segment information.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing instance-level results such as boxes,
            centers, widths, heights, scores, classes, masks, and the number of detected objects.
        """
        instance_data = []
        for seg in segments_info:
            seg_id = seg["id"]
            label_id = seg["label_id"]
            score = seg.get("score", 1.0)

            mask = (panoptic_seg == seg_id).float()
            coords = mask.nonzero()
            if coords.numel() == 0:
                bbox = torch.zeros(4, dtype=torch.float32)
                center = torch.zeros(2, dtype=torch.float32)
                width = torch.tensor(0, dtype=torch.float32)
                height = torch.tensor(0, dtype=torch.float32)
            else:
                y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
                x_min, x_max = coords[:, 1].min(), coords[:, 1].max()
                bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

                center_x = (x_min + x_max) / 2.0
                center_y = (y_min + y_max) / 2.0
                center = torch.tensor([center_x, center_y], dtype=torch.float32)

                width = x_max - x_min
                height = y_max - y_min

            instance_data.append(
                {
                    "mask": mask,
                    "bbox": bbox,
                    "center": center,
                    "width": width,
                    "height": height,
                    "score": torch.tensor(score, dtype=torch.float32),
                    "class_id": torch.tensor(label_id, dtype=torch.long),
                }
            )

        if len(instance_data) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "centers": torch.zeros((0, 2), dtype=torch.float32),
                "widths": torch.zeros((0,), dtype=torch.float32),
                "heights": torch.zeros((0,), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "classes": torch.zeros((0,), dtype=torch.long),
                "masks": torch.zeros((0, *panoptic_seg.shape), dtype=torch.float32),
                "num_objects": 0,
            }

        masks = torch.stack([d["mask"] for d in instance_data])
        boxes = torch.stack([d["bbox"] for d in instance_data])
        centers = torch.stack([d["center"] for d in instance_data])
        widths = torch.stack([d["width"] for d in instance_data])
        heights = torch.stack([d["height"] for d in instance_data])
        scores = torch.stack([d["score"] for d in instance_data])
        classes = torch.stack([d["class_id"] for d in instance_data])

        return {
            "boxes": boxes,
            "centers": centers,
            "widths": widths,
            "heights": heights,
            "scores": scores,
            "classes": classes,
            "masks": masks,
            "num_objects": len(instance_data),
        }


if __name__ == "__main__":
    import os

    extractor = OneFormer_Extractor()
    print("Initialized OneFormer Extractor.")
    extractor.extractor_summary()

    single_image_path = "images/image1.jpg"
    if not os.path.exists(single_image_path):
        raise FileNotFoundError(f"Image not found: {single_image_path}")

    print(f"\nTesting single image prediction for {single_image_path}...")

    pil_image = Image.open(single_image_path).convert("RGB")
    single_output = extractor.predict(pil_image)
    print("Single image prediction result keys:")
    for key, value in single_output.items():
        shape_str = value.shape if isinstance(value, torch.Tensor) else value
        print(f"  {key}: {shape_str}")
        if key == "classes":
            class_names = [
                extractor.labels[int(cls)]
                for cls in value
                if cls < extractor.num_classes
            ]
            print(f"    Class IDs: {value.tolist()}")
            print(f"    Class Names: {class_names}")
