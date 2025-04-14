from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

model.train(
    data="coco.yaml",
    batch=128 * 2,
    device=[3, 4],
    epochs=1,
)
