from ultralytics import YOLO

model = YOLO("d4c-l.yaml")
# model = YOLO("yolo11n.yaml")

model.train(
    data="coco.yaml",
    imgsz=644,
    optimizer="AdamW",
    lr0=1e-3,
    momentum=0.9,
    batch=128 * 2,
    device=[0, 1],
    freeze=[0],
    epochs=1,
    close_mosaic=999,
    project="d4c",
    name="l-coco-adamw-test",
    exist_ok=True,
    fraction=2e-3,
)
