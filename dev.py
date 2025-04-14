from ultralytics import YOLO

model = YOLO("dinov2-yolo11.yaml")
# model = YOLO("yolo11n.yaml")

model.train(
    data="coco.yaml",
    imgsz=644,
    optimizer="AdamW",
    lr0=1e-3,
    momentum=0.9,
    batch=128,
    device=[7],
    freeze=[0],
    epochs=1,
    close_mosaic=999,
    project="dinov2-yolo11",
    name="l-coco-devtest",
    exist_ok=True,
    fraction=2e-3,
)
