from ultralytics import YOLO
 
 
model = YOLO('/home/mjy/ultralytics/yaml/PPAyolov8n-obb.yaml')
# .load('/home/mjy/ultralytics/yolov8n-obb.pt ') # build from YAML and transfer weights

# Train the model
results = model.train(data='/home/mjy/ultralytics/data/coco8.yaml',batch=16,epochs=100)