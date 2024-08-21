import numpy as np
import cv2

label_path = r'/home/mjy/ultralytics/datasets/OBBCrop/labels/test/06144.txt'
image_path = r'/home/mjy/ultralytics/datasets/OBBCrop/image/test/06144.jpg'
            # 定义颜色列表，假设有四个类别  
colors = [  
    [165, 0, 255],       
    [0, 255, 0],        
    [102, 255, 255],       
    [255, 165, 0],      
    [255, 255, 0]      
]

#坐标转换，原始存储的是YOLOv5格式
# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
def xywh2xyxy(x, w1, h1, img):
    labels = ['car','truck','bus','van','freight']
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
    
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型  
    font_scale = 1.6  # 字体大小  
    font_color = colors[int(label)]  # 文本颜色  
    thickness = 4  # 线条粗细  
    
    top_left_x = max(top_left_x, 0)  # 确保文本不会超出图像边界  
    top_left_y= max(top_left_y, 0) 

     
    cv2.putText(img,labels[int(label)], (int(top_left_x), int(top_left_y)-3), font, font_scale, font_color, thickness)  
    # 绘图  rectangle()函数需要坐标为整数
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), font_color, thickness)



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

cv2.imwrite('./GT/irgt.jpg',img)
