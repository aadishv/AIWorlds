from inference import get_model
import eval
import cv2
import os
import numpy as np

model = get_model(model_id="high-stakes-wnyrk/1",
                  api_key=os.environ.get("ROBOFLOW_API_KEY", ""))

print("We have the goods!")


def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Get original dimensions
    original_height, original_width = img.shape[:2]

    # Calculate new dimensions (downscaled)
    new_width = original_width // 100
    new_height = original_height // 100

    # Resize the image to half its original size
    downscaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    _ = model.infer(downscaled_img)
    pass


eval.run_eval(process_image)
