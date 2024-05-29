from ultralytics import YOLO
 
 
model = YOLO('/home/mjy/ultralytics/yaml/PKIICBAMRyolov8n-obb.yaml') # build from YAML and transfer weights

# Train the model
results = model.train(data='/home/mjy/ultralytics/data/coco8.yaml', epochs=100, imgsz=640,resume=False)