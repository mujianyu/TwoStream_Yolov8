#训练
from ultralytics import YOLO
import ultralytics.nn.tasks
model = YOLO('./yaml/Fasteryolov8n.yaml')
results = model.train(data='/home/mjy/ultralytics/data/drone2.yaml',batch=16,epochs=200)
