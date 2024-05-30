from ultralytics import YOLO
 
 
model = YOLO('/home/mjy/ultralytics/yaml/PKICBAMyolov8n-obb.yaml').load('/home/mjy/ultralytics/runs/obb/PKICBAM/weights/best.pt')  # build from YAML and transfer weights


metrics = model.val(data='/home/mjy/ultralytics/data/coco81.yaml',imgsz=640)