# 查看模型信息

from ultralytics import YOLO
model = YOLO('/home/mjy/ultralytics/yaml/xyolov8n.yaml')
model.info()