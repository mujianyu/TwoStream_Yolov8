from ultralytics import YOLO

# Load a model

model = YOLO("/home/mjy/ultralytics/yaml/yolov8n.yaml")


model.export(format='engine',int8=True,dynamic=True,batch=16,data='/home/mjy/ultralytics/pth/baseline.pt')