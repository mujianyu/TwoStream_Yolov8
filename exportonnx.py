from ultralytics import YOLO

# Load a model

model = YOLO("/home/mjy/ultralytics/yaml/3IRyolov8n-obb.yaml").load('/home/mjy/ultralytics/runs/obb/3IR/weights/best.pt')  # load a custom trained model

# model = YOLO('/home/mjy/ultralytics/yaml/NAM.yaml').load('/home/mjy/ultralytics/yolov8n.pt')  

# Export the model
model.export(format='onnx',opset=12)