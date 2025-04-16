from ultralytics import YOLO

model = YOLO("d4c-l.yaml")
# model = YOLO("yolo11n.yaml")

model.train(
    data="coco.yaml",
    imgsz=644,
    optimizer="AdamW",
    lr0=2e-4,
    momentum=0.9,
    batch=64 * 2,
    device=[0, 1],
    freeze=[0],
    epochs=50,
    close_mosaic=5,
    project="d4c",
    name="l-coco-adamw-test",
    exist_ok=True,
)
