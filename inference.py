import os

from PIL import Image

from utils.Draw import draw_and_save_result
from utils.Inferencer import Inferencer


def inference():
    """Perform inference on a single image and a folder of images.

    This function initializes an Inferencer with specified subject, object, and predicate
    values. It then performs inference on a single image, drawing and saving the result if
    the relationship is detected. Finally, it processes all images in a given folder,
    performing inference and saving those images where the relationship exists.

    Returns:
        None
    """
    inferencer = Inferencer(
        subj_class="person",
        obj_class="sky",
        predicate="near",
    )

    print("Performing inference on a single image...")
    image_path = "/Users/neil/Code/LogicVision/images/image6.jpg"
    image = Image.open(image_path)
    result = inferencer.inference_single(image)
    if result.get("exists", True):
        draw_and_save_result(image, result, "single_result.jpg")

    print("Performing inference on a folder of images...")
    folder_path = "/Users/neil/Code/LogicVision/images"
    inferencer.process_folder(folder_path)


if __name__ == "__main__":
    inference()
