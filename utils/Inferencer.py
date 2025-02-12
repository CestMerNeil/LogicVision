import os
from PIL import Image
import argparse

from models.OneFormer_Extractor import OneFormer_Extractor
from models.Logic_Tensor_Networks import Logic_Tensor_Networks
from utils.Draw import draw_and_save_result  # Integrated function for drawing and saving

class Inferencer:
    """
    Inference engine for processing single images and batch processing folders.

    Args:
        subj_class (str): The subject class label (e.g., "person").
        obj_class (str): The object class label (e.g., "tent").
        predicate (str): The relationship predicate (e.g., "near").
        threshold (float): The confidence threshold for determining relationship existence (default: 0.7).
    """
    def __init__(self, subj_class: str, obj_class: str, predicate: str, threshold=0.7):
        self.subj_class = subj_class
        self.obj_class = obj_class
        self.predicate = predicate
        self.threshold = threshold

        # Initialise the OneFormer extractor (only once)
        self.extractor = OneFormer_Extractor()
        self.labels_list = list(self.extractor.model.config.id2label.values())

    def inference_single(self, image: Image.Image) -> dict:
        """
        Performs inference on a single image and returns the result.

        The result dictionary includes subject and object labels for drawing.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            dict: The inference result containing relationship existence, confidence, 
                  bounding box locations, and class labels.
        """
        extractor_result = self.extractor.predict(image)
        ltn_instance = Logic_Tensor_Networks(
            detector_output=extractor_result,
            input_dim=5,
            class_labels=self.labels_list,
            train=False
        )
        result = ltn_instance.inference(self.subj_class, self.obj_class, self.predicate, self.threshold)

        # Add class labels for drawing
        result["subject_class"] = self.subj_class
        result["object_class"] = self.obj_class
        return result

    def process_folder(self, folder_path: str, output_folder: str = "results"):
        """
        Processes all images in a folder, performing inference and saving results.

        Images where the specified relationship exists will be drawn and saved sequentially.

        Args:
            folder_path (str): Path to the folder containing images.
            output_folder (str): Directory to save result images (default: "results").
        """
        os.makedirs(output_folder, exist_ok=True)
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        saved_count = 0

        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(folder_path, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Failed to open image {image_file}: {e}")
                continue

            result = self.inference_single(image)
            if result.get("exists", False):
                filename = f"image_{saved_count:03d}.jpg"
                draw_and_save_result(image, result, filename)  # Only save if the relationship exists
                saved_count += 1
                print(f"Saved image: {filename}")
            else:
                print(f"No detected relationship in {image_file}.")

        print(f"Total saved images: {saved_count}.")