from ultralytics import YOLO
 
 
model = YOLO('/home/mjy/ultralytics/yaml/GCBAMyolov8n-obb.yaml').load('/home/mjy/ultralytics/runs/obb/CBAMSA/weights/best.pt')  # build from YAML and transfer weights


metrics = model.val(data='/home/mjy/ultralytics/coco8.yaml',imgsz=640)