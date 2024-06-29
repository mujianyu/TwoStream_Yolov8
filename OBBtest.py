from ultralytics import YOLO
 
 
model = YOLO('/home/mjy/ultralytics/runs/obb/train17/weights/best.pt')  # build from YAML and transfer weights


metrics = model.val(data='/home/mjy/ultralytics/data/coco81.yaml',imgsz=864)