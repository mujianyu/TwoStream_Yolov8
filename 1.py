import cv2
import matplotlib.pyplot as plt
img = cv2.imread('/home/mjy/ultralytics/datasets/rgbir/image/train/00001.jpg') #读取
cv2.imshow( "11",img) #展示
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转化为i灰度图像
cv2.imwrite("/home/mjy/ultralytics/q.jpg", gray) #保存