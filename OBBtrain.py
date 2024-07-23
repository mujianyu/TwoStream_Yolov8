from ultralytics import YOLO
import ultralytics.nn.tasks
 
model = YOLO('./yaml/Fasteryolov8n.yaml')

# Train the model
results = model.train(data='/home/mjy/ultralytics/data/drone.yaml',batch=16,epochs=1,amp=False,half=False)
