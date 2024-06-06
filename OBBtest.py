from ultralytics import YOLO
 
 
model = YOLO('/home/mjy/ultralytics/yaml/PKIADyolov8n-obb.yaml').load('/home/mjy/ultralytics/runs/obb/PKIAD/weights/best.pt')  # build from YAML and transfer weights


metrics = model.val(data='/home/mjy/ultralytics/data/coco81.yaml',imgsz=640)