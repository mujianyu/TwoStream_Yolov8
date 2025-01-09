#训练
from multiprocessing import freeze_support
from ultralytics import YOLO

import ultralytics.nn.tasks
def main():
    model = YOLO(r'D:\Doctor\TwoStream_Yolov8-main\yaml\PC2f_MPF_yolov8n.yaml')
    results = model.train(data=r'D:\Doctor\TwoStream_Yolov8-main/data/drone2.yaml',batch=8,epochs=200)


if __name__ == "__main__":
    freeze_support()  # 加上这一句,防止windows环境下的多进程报错
    main()
