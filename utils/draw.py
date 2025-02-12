import os
from PIL import Image, ImageDraw, ImageFont

def draw_and_save_result(image: Image.Image, result: dict, filename: str) -> Image.Image:
    """
    Draws bounding boxes and labels on the image, and saves it if the relationship exists.

    Args:
        image (PIL.Image.Image): The input image.
        result (dict): The inference result containing:
            - "exists" (bool): Whether the relationship is detected.
            - "subject_locations" (dict): Contains "centers", "widths", and "heights" for subjects.
            - "object_locations" (dict): Contains "centers", "widths", and "heights" for objects.
            - "subject_class" (str): The label of the subject.
            - "object_class" (str): The label of the object.
        filename (str): The name of the output file.

    Returns:
        PIL.Image.Image: The image with bounding boxes and labels drawn.
    """
    if not result.get("exists", False):
        print("Relationship does not exist. Image will not be saved.")
        return image

    draw = ImageDraw.Draw(image)
    subj_label = result.get("subject_class", "subject")
    obj_label = result.get("object_class", "object")
    
    # Load font, fallback to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

    # Draw subject bounding boxes (red)
    subj = result.get("subject_locations", {})
    for center, width, height in zip(subj.get("centers", []),
                                     subj.get("widths", []),
                                     subj.get("heights", [])):
        cx, cy = center
        left, top = cx - width / 2, cy - height / 2
        right, bottom = cx + width / 2, cy + height / 2
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)
        draw.text((left, top), subj_label, fill="red", font=font)

    # Draw object bounding boxes (blue)
    obj = result.get("object_locations", {})
    for center, width, height in zip(obj.get("centers", []),
                                     obj.get("widths", []),
                                     obj.get("heights", [])):
        cx, cy = center
        left, top = cx - width / 2, cy - height / 2
        right, bottom = cx + width / 2, cy + height / 2
        draw.rectangle([(left, top), (right, bottom)], outline="blue", width=2)
        draw.text((left, top), obj_label, fill="blue", font=font)

    # Save the processed image
    os.makedirs("results", exist_ok=True)
    file_path = os.path.join("results", filename)
    image.save(file_path)
    print(f"Image saved to {file_path}")
    
    return image

if __name__ == "__main__":
    from PIL import Image

    result = {
        'exists': True,
        'confidence': 1.0,
        'message': 'Inference successful',
        'subject_locations': {
            'centers': [[429.5, 421.0], [352.5, 304.0], [108.5, 283.0]],
            'widths': [41, 165, 17],
            'heights': [56, 66, 24]
        },
        'object_locations': {
            'centers': [[449.5, 422.0]],
            'widths': [899],
            'heights': [292]
        },
        'subject_class': 'person',
        'object_class': 'tent',
        'predicate': 'near'
    }
    
    image = Image.open("/Users/neil/Code/LogicVision/images/image0.jpg")
    image_with_boxes = draw_and_save_result(image, result, "result.jpg")