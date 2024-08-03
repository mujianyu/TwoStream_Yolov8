from ultralytics import YOLO
 




model = YOLO('/home/mjy/ultralytics/pth/p.pt') 

model.predict(source=r'/home/mjy/ultralytics/datasets/OBBCrop/images/test/00036.jpg', save=True, show=False, imgsz=640)
