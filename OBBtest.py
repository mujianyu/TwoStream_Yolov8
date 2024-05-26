from ultralytics import YOLO
 
 
model = YOLO('/home/mjy/ultralytics/yaml/SACBAMyolov8n-obb.yaml').load('/home/mjy/ultralytics/runs/obb/SACBAM/weights/best.pt')  # build from YAML and transfer weights


metrics = model.val(data='/home/mjy/ultralytics/coco81.yaml',imgsz=640)