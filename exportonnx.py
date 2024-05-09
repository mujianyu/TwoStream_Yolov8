from ultralytics import YOLO

# Load a model

model = YOLO('/home/mjy/ultralytics/runs/detect/3IR/weights/best.pt')  # load a custom trained model

# Export the model
model.export(format='onnx')