import os

from PIL import Image, ImageDraw, ImageFont


def draw_and_save_result(
    image: Image.Image, result: dict, filename: str, output_folder: str = "results"
) -> Image.Image:
    """Draw bounding boxes and labels on an image and save it if a relationship exists.

    Args:
        image (Image.Image): The input image.
        result (dict): Inference results containing the following keys:
            - "exists" (bool): Indicates whether the relationship exists in the image.
            - "subject_locations" (dict): Contains "centers", "widths", and "heights" for subjects.
            - "object_locations" (dict): Contains "centers", "widths", and "heights" for objects.
            - "subject_class" (str): Label for the subject.
            - "object_class" (str): Label for the object.
            - "individual_scores" (list): Scores for each subject-object pair.
            - "predicate" (str): The relationship predicate.
        filename (str): The name of the output file to save the processed image.

    Returns:
        Image.Image: The image with bounding boxes and labels drawn.
    """
    # Check if there are any matching pairs instead of using the global exists flag
    individual_scores = result.get("individual_scores", [])
    threshold = 0.7  # Using standard threshold

    # If no individual scores, fall back to global exists flag
    if not individual_scores:
        has_matching_pairs = result.get("exists", False)
    else:
        has_matching_pairs = any(
            any(score >= threshold for score in row) for row in individual_scores
        )

    if not has_matching_pairs:
        print("No matching relationships found. Image will not be saved.")
        return image

    draw = ImageDraw.Draw(image)
    subj_label = result.get("subject_class", "subject")
    obj_label = result.get("object_class", "object")
    predicate = result.get("predicate", "")

    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

    # Get locations
    subj = result.get("subject_locations", {})
    obj = result.get("object_locations", {})

    subjects_drawn = set()
    objects_drawn = set()

    # Only draw objects that have high relationship scores
    if individual_scores:
        for i, subj_row in enumerate(individual_scores):
            for j, score in enumerate(subj_row):
                if score >= threshold:
                    # Draw this subject-object pair since it satisfies the relationship
                    if i < len(subj.get("centers", [])) and j < len(
                        obj.get("centers", [])
                    ):
                        # Draw subject if not already drawn
                        if i not in subjects_drawn:
                            center = subj.get("centers", [])[i]
                            width = subj.get("widths", [])[i]
                            height = subj.get("heights", [])[i]

                            cx, cy = center
                            left = cx - width / 2
                            top = cy - height / 2
                            right = cx + width / 2
                            bottom = cy + height / 2
                            draw.rectangle(
                                [(left, top), (right, bottom)], outline="red", width=2
                            )
                            draw.text(
                                (left, top),
                                f"{subj_label} {i+1}",
                                fill="red",
                                font=font,
                            )
                            subjects_drawn.add(i)

                        # Draw object if not already drawn
                        if j not in objects_drawn:
                            center = obj.get("centers", [])[j]
                            width = obj.get("widths", [])[j]
                            height = obj.get("heights", [])[j]

                            cx, cy = center
                            left = cx - width / 2
                            top = cy - height / 2
                            right = cx + width / 2
                            bottom = cy + height / 2
                            draw.rectangle(
                                [(left, top), (right, bottom)], outline="blue", width=2
                            )
                            draw.text(
                                (left, top),
                                f"{obj_label} {j+1}",
                                fill="blue",
                                font=font,
                            )
                            objects_drawn.add(j)

                        # Draw a line connecting the related objects
                        subj_center = subj.get("centers", [])[i]
                        obj_center = obj.get("centers", [])[j]
                        draw.line(
                            [tuple(subj_center), tuple(obj_center)],
                            fill="green",
                            width=1,
                        )

                        # Add relation label with score
                        mid_x = (subj_center[0] + obj_center[0]) / 2
                        mid_y = (subj_center[1] + obj_center[1]) / 2
                        draw.text(
                            (mid_x, mid_y),
                            f"{predicate}: {score:.2f}",
                            fill="green",
                            font=font,
                        )
    else:
        # Fall back to original behavior if no individual scores available
        # Draw all subject locations (red)
        for i, (center, width, height) in enumerate(
            zip(
                subj.get("centers", []), subj.get("widths", []), subj.get("heights", [])
            )
        ):
            cx, cy = center
            left = cx - width / 2
            top = cy - height / 2
            right = cx + width / 2
            bottom = cy + height / 2
            draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)
            draw.text((left, top), f"{subj_label} {i+1}", fill="red", font=font)

        # Draw all object locations (blue)
        for i, (center, width, height) in enumerate(
            zip(obj.get("centers", []), obj.get("widths", []), obj.get("heights", []))
        ):
            cx, cy = center
            left = cx - width / 2
            top = cy - height / 2
            right = cx + width / 2
            bottom = cy + height / 2
            draw.rectangle([(left, top), (right, bottom)], outline="blue", width=2)
            draw.text((left, top), f"{obj_label} {i+1}", fill="blue", font=font)

    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)
    image.save(file_path)
    print(f"Image saved to {file_path}")

    return image


if __name__ == "__main__":
    from PIL import Image

    result = {
        "exists": True,
        "confidence": 1.0,
        "message": "Inference successful",
        "subject_locations": {
            "centers": [[429.5, 421.0], [352.5, 304.0], [108.5, 283.0]],
            "widths": [41, 165, 17],
            "heights": [56, 66, 24],
        },
        "object_locations": {
            "centers": [[449.5, 422.0]],
            "widths": [899],
            "heights": [292],
        },
        "subject_class": "person",
        "object_class": "tent",
        "predicate": "near",
    }

    image = Image.open("/Users/neil/Code/LogicVision/images/image0.jpg")
    image_with_boxes = draw_and_save_result(image, result, "result.jpg")
