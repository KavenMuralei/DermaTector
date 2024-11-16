from ultralytics import YOLO
import cv2

# Loads the pre trained model. In our case this is YOLOv8
# model = YOLO('tolov8n.pt')

model = YOLO("yolo11n.pt")  # load an official model


# Predict with the model
results = model(source="0", show=True, conf=0.4, save=True)  # predict on an image