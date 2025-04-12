from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(
    data="coco128.yaml",
    device=[1, 2],
    batch=32,
    optimizer="AdamW",
    momentum=0.9,
    project="comet-example-yolo11",
    name="2gpu-32bs",
    epochs=1,
)

# 1 gpu 32 bs: 4/2
# 2 gpu 32 bs: 4/4
