from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

model.val(data="coco128.yaml", batch=32, device=[0, 1])
