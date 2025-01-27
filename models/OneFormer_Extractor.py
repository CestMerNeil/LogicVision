import torch
import pprint
import tomllib
from PIL import Image
from typing import Dict, List, Tuple

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
        Print the summary of the OneFormer Extractor
        """
        print(f"Processor: {self.processor}")
        print(f"Model: {self.model}")
        print(f"Labels: {self.labels}")
        print(f"Number of Classes: {self.num_classes}")

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict the image using the OneFormer model.
        Args:
            image (torch.Tensor): Image of shape [3, H, W]
        Returns:
            Dict[str, torch.Tensor]
        """
        pil_image = self._to_pil(image)
        inputs = self.processor(
            images=pil_image,
            task_inputs=[config['OneFormer_Extractor']['task_inputs']],
            return_tensors="pt"
        )
        outputs = self.model(**inputs)

        return self._process_result(outputs, image_size=image.shape[1:])

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict the images using the OneFormer model and align the results for batch processing.
        Args:
            images (torch.Tensor): (B, 3, H, W)
        Returns:
            Dict[str, torch.Tensor]
        """
        B, C, H, W = images.shape
        pil_images = [self._to_pil(images[i]) for i in range(B)]

        inputs = self.processor(
            images=pil_images,
            task_inputs=[config['OneFormer_Extractor']['task_inputs']] * B,
            return_tensors="pt"
        )
        outputs = self.model(**inputs)

        processed_list = self.processor.post_process_panoptic_segmentation(
            outputs,
            threshold=self.conf_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[(H, W)] * B
        )

        batch_results = []
        max_num_objs = 0
        for i in range(B):
            single_res = self._panoptic_to_instance(
                processed_list[i]["segmentation"],
                processed_list[i]["segments_info"]
            )
            batch_results.append(single_res)
            if single_res["num_objects"] > max_num_objs:
                max_num_objs = single_res["num_objects"]

        aligned_results = {
            "boxes": torch.zeros(B, max_num_objs, 4),
            "centers": torch.zeros(B, max_num_objs, 2),
            "widths": torch.zeros(B, max_num_objs),
            "heights": torch.zeros(B, max_num_objs),
            "scores": torch.zeros(B, max_num_objs),
            "classes": torch.zeros(B, max_num_objs),
            "masks": torch.zeros(B, max_num_objs, H, W),
            "num_objects": torch.zeros(B, dtype=torch.long),
            "image_size": torch.tensor([H, W], dtype=torch.long),
        }

        for i, res in enumerate(batch_results):
            n_i = res["num_objects"]
            aligned_results["num_objects"][i] = n_i
            if n_i > 0:
                aligned_results["masks"][i, :n_i] = res["masks"]
                aligned_results["boxes"][i, :n_i] = res["boxes"]
                aligned_results["centers"][i, :n_i] = res["centers"]
                aligned_results["widths"][i, :n_i] = res["widths"]
                aligned_results["heights"][i, :n_i] = res["heights"]
                aligned_results["scores"][i, :n_i] = res["scores"]
                aligned_results["classes"][i, :n_i] = res["classes"]

        return aligned_results

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

        instance_res = self._panoptic_to_instance(panoptic_seg, segments_info)

        instance_res["image_size"] = torch.tensor([H, W], dtype=torch.long)
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
                "masks": torch.zeros((0, *panoptic_seg.shape), dtype=torch.float32),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "centers": torch.zeros((0, 2), dtype=torch.float32),
                "widths": torch.zeros((0,), dtype=torch.float32),
                "heights": torch.zeros((0,), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "classes": torch.zeros((0,), dtype=torch.long),
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

    def _to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert the tensor to PIL Image
        Args:
            tensor (torch.Tensor): Image tensor
        Returns:
            Image: PIL Image
        """
        tensor = tensor.detach().cpu()
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        arr = tensor.byte().numpy().transpose(1, 2, 0)
        return Image.fromarray(arr)


if __name__ == "__main__":
    import os
    import torch

    extractor = OneFormer_Extractor()
    print("Initialized OneFormer Extractor.")
    extractor.extractor_summary()

    single_image_path = "images/image1.jpg"
    if not os.path.exists(single_image_path):
        raise FileNotFoundError(f"Image not found: {single_image_path}")

    print(f"\nTesting single image prediction for {single_image_path}...")

    single_image_tensor = torch.rand(3, 640, 640)
    #single_image_tensor = Image.open(single_image_path)
    single_output = extractor.predict(single_image_tensor)
    print("Single image prediction result keys:")
    for key, value in single_output.items():
        shape_str = value.shape if isinstance(value, torch.Tensor) else value
        print(f"  {key}: {shape_str}")
        if key == "classes":
            class_names = [extractor.labels[int(cls)] for cls in value if cls < extractor.num_classes]
            print(f"    Class IDs: {value.tolist()}")
            print(f"    Class Names: {class_names}")

    batch_images = torch.rand(4, 3, 640, 640)
    print("\nTesting batch image prediction...")
    batch_output = extractor.predict_batch(batch_images)
    print("Batch prediction result keys:")
    for key, value in batch_output.items():
        print(f"  {key}: {value.shape if isinstance(value, torch.Tensor) else value}")
        if key == "classes":
            for i in range(batch_output["classes"].shape[0]):
                class_ids = batch_output["classes"][i].tolist()
                class_names = [
                    extractor.labels[int(cls)]
                    for cls in class_ids
                    if cls < extractor.num_classes
                ]
                print(f"    Image {i + 1}:")
                print(f"      Class IDs: {class_ids}")
                print(f"      Class Names: {class_names}")