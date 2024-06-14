
 


from ultralytics import YOLO
 
 
model = YOLO('/home/mjy/ultralytics/yaml/yolov8n.yaml') # build from YAML and transfer weights

# Train the model
results = model.train(data='/home/mjy/ultralytics/data/vedai.yaml', epochs=120, imgsz=1024)