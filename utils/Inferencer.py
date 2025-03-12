import argparse
import os

from PIL import Image

from models.Logic_Tensor_Networks import Logic_Tensor_Networks
from models.OneFormer_Extractor import OneFormer_Extractor
from models.YOLO_Extractor import YOLO_Extractor
from utils.Draw import draw_and_save_result


class Inferencer:
    """Inference engine for processing single images and batch processing folders.

    This class performs inference using a specified extractor and a logic tensor network,
    allowing both single image and folder batch processing.

    Attributes:
        subj_class (str): The subject class label (e.g., "person").
        obj_class (str): The object class label (e.g., "tent").
        predicate (str): The relationship predicate (e.g., "near").
        threshold (float): The confidence threshold for determining relationship existence.
        extractor (OneFormer_Extractor or YOLO_Extractor): The chosen extractor for object detection.
        labels_list (list): List of class labels from the extractor.
    """

    def __init__(
        self,
        subj_class: str,
        obj_class: str,
        predicate: str,
        threshold: float = 0.7,
        extractor: str = "OneFormer",
    ):
        """Initializes the Inferencer with the specified parameters.

        Args:
            subj_class (str): The subject class label.
            obj_class (str): The object class label.
            predicate (str): The relationship predicate.
            threshold (float, optional): Confidence threshold for relationship existence. Defaults to 0.7.
            extractor (str, optional): Type of extractor to use ("OneFormer" or "YOLO"). Defaults to "OneFormer".

        Raises:
            ValueError: If an invalid extractor type is provided.
        """
        self.subj_class = subj_class
        self.obj_class = obj_class
        self.predicate = predicate
        self.threshold = threshold

        if extractor == "OneFormer":
            self.extractor = OneFormer_Extractor()
        elif extractor == "YOLO":
            self.extractor = YOLO_Extractor()
        else:
            raise ValueError(f"Invalid extractor type: {extractor}")

        self.labels_list = list(self.extractor.labels.values())

    def inference_single(self, image: Image.Image) -> dict:
        """Performs inference on a single image and returns the result.

        The method uses the chosen extractor to predict object detections, then applies a logic
        tensor network to evaluate the specified relationship between subject and object.

        Args:
            image (Image.Image): The input image for inference.

        Returns:
            dict: Inference results containing relationship existence, confidence, bounding box locations,
                  and class labels.
        """
        extractor_result = self.extractor.predict(image)
        ltn_instance = Logic_Tensor_Networks(
            detector_output=extractor_result,
            input_dim=5,
            class_labels=self.labels_list,
            train=False,
        )
        result = ltn_instance.inference(
            self.subj_class, self.obj_class, self.predicate, self.threshold
        )
        result["subject_class"] = self.subj_class
        result["object_class"] = self.obj_class
        return result

    def process_folder(self, folder_path: str, output_folder: str = "results"):
        """Processes all images in a folder, performs inference, and saves the results.

        The method iterates over all supported image files in the specified folder,
        performs inference on each image, and saves those images where at least one
        subject-object pair meets the relationship criteria using soft aggregation.

        Args:
            folder_path (str): Path to the folder containing images.
            output_folder (str, optional): Directory to save result images. Defaults to "results".
        """
        os.makedirs(output_folder, exist_ok=True)
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = [
            f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)
        ]
        saved_count = 0

        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(folder_path, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Failed to open image {image_file}: {e}")
                continue

            result = self.inference_single(image)

            # Use the exists field which is now based on soft aggregation
            if result.get("exists", False):
                filename = f"image_{saved_count:03d}.jpg"
                draw_and_save_result(image, result, filename, output_folder)
                saved_count += 1
                print(
                    f"Saved image: {filename} - soft score: {result.get('soft_score', 0)}"
                )
            else:
                print(
                    f"No matching relationships in {image_file} - soft score: {result.get('soft_score', 0)}"
                )

        print(f"Total saved images: {saved_count}.")
