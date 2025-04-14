from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

model.val(
    data="coco.yaml",
    batch=32 * 2,
    device=[3, 4],
    epochs=1,
)
