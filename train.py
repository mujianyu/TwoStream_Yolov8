from ultralytics import YOLO
 
 
# 加载模型
model = YOLO('/home/mjy/ultralytics/yaml/NAM.yaml').load('/home/mjy/ultralytics/yolov8n.pt')  # 从YAML构建并转移权重
 
if __name__ == '__main__':
    # 训练模型
    #results = model.train(data='/home/mjy/ultralytics/coco8.yaml', epochs=100, imgsz=640)
    metrics = model.val(data='/home/mjy/ultralytics/coco8.yaml',imgsz=640)
