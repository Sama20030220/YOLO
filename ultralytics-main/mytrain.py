from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("/root/5/ultralytics-main/ultralytics/cfg/models/v8/myyolov8.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("/root/5/ultralytics-main/best.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="/root/5/ultralytics-main/ultralytics/cfg/datasets/mycoco128.yaml", epochs=100)

