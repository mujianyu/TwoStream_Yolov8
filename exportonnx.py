from ultralytics import YOLO

# Load a model

model = YOLO("/home/mjy/ultralytics/yaml/yolov8n.yaml").load('/home/mjy/ultralytics/runs/detect/train41/weights/best.pt')  # load a custom trained model

# model = YOLO('/home/mjy/ultralytics/yaml/NAM.yaml').load('/home/mjy/ultralytics/yolov8n.pt')  

# Export the model
model.export(format='onnx',opset=12)