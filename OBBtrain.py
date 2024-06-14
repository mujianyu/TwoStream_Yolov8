from ultralytics import YOLO
 
 
model = YOLO('/home/mjy/ultralytics/yaml/3IRyolov8n-obb.yaml') # build from YAML and transfer weights

# Train the model
results = model.train(data='/home/mjy/ultralytics/data/coco8.yaml', epochs=12, imgsz=1024)