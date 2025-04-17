from inference import get_model
import supervision as sv
import cv2

# Load model
model = get_model(model_id="high-stakes-wnyrk/1")

# Load image with cv2
image = cv2.imread("saved_images/blob_detection_rgb_20250416_202215.jpg")

# Run inference
results = model.infer(image)[0]
# print(results)
# exit()
# Load results into Supervision Detection API
detections = sv.Detections.from_inference(results)

# Create Supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Extract labels array from inference results
print(results)
labels = [p.class_name for p in results.predictions]


# Apply results to image using Supervision annotators
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels
)

# Write annotated image to file or display image
sv.plot_image(annotated_image)
