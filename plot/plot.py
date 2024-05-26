import numpy as np
import cv2
import torch

label_path = r'/home/mjy/ultralytics/datasets/rgbir/labels/train/00002.txt'
image_path = r'/home/mjy/ultralytics/datasets/rgbir/image/train/00002.jpg'

#坐标转换，原始存储的是YOLOv5格式
# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
def xywh2xyxy(x, w1, h1, img):
    labels = ['van','car','truck','bus','freight car']
    label, x, y, w, h = x
    print("原图宽高:\nw1={}\nh1={}".format(w1, h1))
    #边界框反归一化
    x_t = x*w1
    y_t = y*h1
    w_t = w*w1
    h_t = h*h1
    print("反归一化后输出：\n第一个:{}\t第二个:{}\t第三个:{}\t第四个:{}\t\n\n".format(x_t,y_t,w_t,h_t))

    #计算坐标
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2
    print('标签:{}'.format(labels[int(label)]))
    print("左上x坐标:{}".format(top_left_x))
    print("左上y坐标:{}".format(top_left_y))
    print("右下x坐标:{}".format(bottom_right_x))
    print("右下y坐标:{}".format(bottom_right_y))

    # 绘图  rectangle()函数需要坐标为整数
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (0, 255, 0), 2)



#读取 labels
with open(label_path, 'r') as f:
    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
    print(lb)

# 读取图像文件
img = cv2.imread(str(image_path))
h, w = img.shape[:2]

for x in lb:
    # 反归一化并得到左上和右下坐标，画出矩形框
    xywh2xyxy(x, w, h, img)

cv2.imshow('show', img)
cv2.imwrite('11.png',img)
cv2.waitKey(0)  # 按键结束
cv2.destroyAllWindows()
