from models.OneFormer_Extractor import OneFormer_Extractor
from models.Logic_Tensor_Networks import Logic_Tensor_Networks
from PIL import Image
from utils.Draw import draw_bounding_boxes, save_result_image

def inference(image: Image.Image, subj_class: str, obj_class: str, predicate: str, threshold=0.7):
    extractor = OneFormer_Extractor()  # 调用类构造函数
    extractor_result = extractor.predict(image)

    labels_list = list(extractor.model.config.id2label.values())

    ltn_instance = Logic_Tensor_Networks(
        detector_output=extractor_result,
        input_dim=5,
        class_labels=labels_list,
        train=False
    )

    result = ltn_instance.inference(subj_class, obj_class, predicate, threshold)
    return result

if __name__ == "__main__":
    import os

    single_image_path = "images/image4.jpg"
    if not os.path.exists(single_image_path):
        raise FileNotFoundError(f"Image not found: {single_image_path}")

    # 直接使用 PIL Image 作为输入
    pil_image = Image.open(single_image_path).convert("RGB")
    result = inference(pil_image, "person", "sky", "near")
    image_with_boxes = draw_bounding_boxes(pil_image, result)
    os.makedirs("results", exist_ok=True)
    save_result_image(image_with_boxes, "image0_result.jpg")
    print(result)