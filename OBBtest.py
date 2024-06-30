from ultralytics import YOLO
 
 
model = YOLO('/home/mjy/ultralytics/runs/detect/train44/weights/best.pt')  # build from YAML and transfer weights


metrics = model.val(data='/home/mjy/ultralytics/data/drone2.yaml',imgsz=640)