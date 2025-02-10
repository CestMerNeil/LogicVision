import os
from PIL import Image, ImageDraw

def draw_bounding_boxes(image: Image.Image, result: dict) -> Image.Image:
    """
    Draw bounding boxes for subjects and objects on the input image.
    
    Args:
        image (PIL.Image.Image): The input image.
        result (dict): Detection results with keys:
            - "subject_locations": {"centers": [[x, y], ...], "widths": [w, ...], "heights": [h, ...]}
            - "object_locations": {"centers": [[x, y], ...], "widths": [w, ...], "heights": [h, ...]}
    
    Returns:
        PIL.Image.Image: The image with drawn bounding boxes.
    """
    draw = ImageDraw.Draw(image)

    # Draw subject bounding boxes in red.
    subj = result.get("subject_locations", {})
    for center, width, height in zip(subj.get("centers", []),
                                      subj.get("widths", []),
                                      subj.get("heights", [])):
        cx, cy = center
        left   = cx - width / 2
        top    = cy - height / 2
        right  = cx + width / 2
        bottom = cy + height / 2
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)

    # Draw object bounding boxes in blue.
    obj = result.get("object_locations", {})
    for center, width, height in zip(obj.get("centers", []),
                                      obj.get("widths", []),
                                      obj.get("heights", [])):
        cx, cy = center
        left   = cx - width / 2
        top    = cy - height / 2
        right  = cx + width / 2
        bottom = cy + height / 2
        draw.rectangle([(left, top), (right, bottom)], outline="blue", width=2)

    return image

def save_result_image(image: Image.Image, filename: str):
    """
    Save the image with drawn bounding boxes to the 'results' folder.
    
    Args:
        image (PIL.Image.Image): The image to be saved.
        filename (str): The filename to use for saving the image.
    """
    os.makedirs("results", exist_ok=True)
    file_path = os.path.join("results", filename)
    image.save(file_path)
    print(f"Image saved to {file_path}")

# Example usage:
# from PIL import Image
# result = {
#     'subject_locations': {
#         'centers': [[611.5, 369.5], [641.0, 752.5], [726.0, 383.0]],
#         'widths': [95, 154, 146],
#         'heights': [307, 173, 294]
#     },
#     'object_locations': {
#         'centers': [[639.5, 253.5]],
#         'widths': [1279],
#         'heights': [507]
#     }
# }
# image = Image.open("path/to/your/image.jpg")
# image_with_boxes = draw_bounding_boxes(image, result)
# save_result_image(image_with_boxes, "result.jpg")