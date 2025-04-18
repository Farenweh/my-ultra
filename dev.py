from ultralytics import YOLO

model = YOLO("d4c-l-reg.yaml")
model.train(
    data="coco.yaml",
    imgsz=644,
    device=[0, 2],
    optimizer="AdamW",
    freeze=[0],
    batch=48 * 2,
    lr0=1e-3,
    momentum=0.9,
    epochs=50,
    project="d4c",
    name="d4c-l-reg-coco",
)
