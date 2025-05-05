from inference import get_model
import eval
import cv2
import os

model = get_model(model_id="high-stakes-wnyrk/1",
                  api_key=os.environ.get("ROBOFLOW_API_KEY", ""))

print("We have the goods!")


def process_image(image_path):
    _ = model.infer(cv2.imread(image_path))
    pass


eval.run_eval(process_image)
