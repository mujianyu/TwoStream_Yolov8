# 测试
from ultralytics import YOLO 
model = YOLO('/home/mjy/ultralytics/pth/baseline.pt') 
metrics = model.val(data='/home/mjy/ultralytics/data/drone2.yaml',split='test',imgsz=640,batch=16)
