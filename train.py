

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # yolov8n
    # yolov8-C2f-DEConv.yaml
    # yolov8-C2f-fadc.yaml
    # yolov8-pkinet.yaml
    # yolov8-C2f-PKI.yaml
    # yolov8-dyhead-DCNV3 .yaml
    #yolov8-C2f-Faster-CGLU.yamldasdsadasdsaqa wrmj.lik
    # /home/mjy/ultralytics-main/ultralytics/cfg/models/v8/yolov8-Faster.yaml
    model = YOLO('/home/mjy/ultralytics/yaml/yolov8n-CSFM.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/home/mjy/ultralytics/data/vedai.yaml',
                cache=False,
                imgsz=512,
                epochs=400,
                batch=32,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                patience=0, # close earlystop
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )