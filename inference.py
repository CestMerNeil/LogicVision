import os
from PIL import Image
from utils.Inferencer import Inferencer
from utils.Draw import draw_and_save_result

def inference():
    inferencer = Inferencer(
        subj_class="person",
        obj_class="sky",
        predicate="near",
    )

    # Single image inference
    print("Performing inference on a single image...")
    image_path = "/Users/neil/Code/LogicVision/images/image6.jpg"
    image = Image.open(image_path)
    result = inferencer.inference_single(image)
    if result.get("exists", True):
        draw_and_save_result(image, result, "single_result.jpg")

    # Folder inference
    print("Performing inference on a folder of images...")
    folder_path = "/Users/neil/Code/LogicVision/images"
    inferencer.process_folder(folder_path)

if __name__ == "__main__":
    inference()