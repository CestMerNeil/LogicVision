import torch
import pprint
import tomllib
from PIL import Image
from typing import Dict, List

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

with open("config.toml", "rb") as config_file:
    config = tomllib.load(config_file)

pprint.pprint(f"Config List: {config}")

class OneFormer_Extractor:
    def __init__(self):
        self.processor = OneFormerProcessor.from_pretrained(
            config['OneFormer_Extractor']['processor'],
            # cache_dir=config['OneFormer_Extractor']['cache_dir']
        )
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            config['OneFormer_Extractor']['model'],
            # cache_dir=config['OneFormer_Extractor']['cache_dir']
        )

        self.conf_threshold = config['OneFormer_Extractor']['conf_threshold']
        self.mask_threshold = config['OneFormer_Extractor']['mask_threshold']

        self.labels = self.model.config.id2label
        self.num_classes = len(self.labels)

    def extractor_summary(self):
        """
        Print the summary of the OneFormer Extractor.
        """
        print(f"Processor: {self.processor}")
        print(f"Model: {self.model}")
        print(f"Labels: {self.labels}")
        print(f"Number of Classes: {self.num_classes}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Predict the objects in a single image using the OneFormer model.
        The input is in PIL Image format.
        
        Args:
            image (PIL.Image.Image): The input image.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the detection results:
                "boxes", "centers", "widths", "heights", "scores", "classes", "masks", "num_objects"
        """
        # Directly use the input PIL Image without conversion.
        inputs = self.processor(
            images=image,
            task_inputs=[config['OneFormer_Extractor']['task_inputs']],
            return_tensors="pt"
        )
        outputs = self.model(**inputs)

        # PIL Image size is (width, height); reverse it to (height, width)
        return self._process_result(outputs, image_size=image.size[::-1])

    def _process_result(self, outputs, image_size: tuple) -> Dict[str, torch.Tensor]:
        H, W = image_size
        processed = self.processor.post_process_panoptic_segmentation(
            outputs,
            threshold=self.conf_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[(H, W)]
        )[0]

        panoptic_seg = processed["segmentation"]
        segments_info = processed["segments_info"]

        # Return a dictionary containing only the detected object information.
        instance_res = self._panoptic_to_instance(panoptic_seg, segments_info)
        return instance_res

    def _panoptic_to_instance(self, panoptic_seg: torch.Tensor, segments_info: List[dict]) -> Dict[str, torch.Tensor]:
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

            instance_data.append({
                "mask": mask,
                "bbox": bbox,
                "center": center,
                "width": width,
                "height": height,
                "score": torch.tensor(score, dtype=torch.float32),
                "class_id": torch.tensor(label_id, dtype=torch.long)
            })

        if len(instance_data) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "centers": torch.zeros((0, 2), dtype=torch.float32),
                "widths": torch.zeros((0,), dtype=torch.float32),
                "heights": torch.zeros((0,), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "classes": torch.zeros((0,), dtype=torch.long),
                "masks": torch.zeros((0, *panoptic_seg.shape), dtype=torch.float32),
                "num_objects": 0
            }

        masks   = torch.stack([d["mask"] for d in instance_data])
        boxes   = torch.stack([d["bbox"] for d in instance_data])
        centers = torch.stack([d["center"] for d in instance_data])
        widths  = torch.stack([d["width"] for d in instance_data])
        heights = torch.stack([d["height"] for d in instance_data])
        scores  = torch.stack([d["score"] for d in instance_data])
        classes = torch.stack([d["class_id"] for d in instance_data])

        return {
            "boxes": boxes,
            "centers": centers,
            "widths": widths,
            "heights": heights,
            "scores": scores,
            "classes": classes,
            "masks": masks,
            "num_objects": len(instance_data)
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

    # Directly use a PIL Image as input.
    pil_image = Image.open(single_image_path).convert("RGB")
    single_output = extractor.predict(pil_image)
    print("Single image prediction result keys:")
    for key, value in single_output.items():
        shape_str = value.shape if isinstance(value, torch.Tensor) else value
        print(f"  {key}: {shape_str}")
        if key == "classes":
            class_names = [extractor.labels[int(cls)] for cls in value if cls < extractor.num_classes]
            print(f"    Class IDs: {value.tolist()}")
            print(f"    Class Names: {class_names}")