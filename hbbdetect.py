import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from tqdm import tqdm
def list_all_files(startpath):  
    all_files = []  
    
    for root, dirs, files in os.walk(startpath):  
        for name in files:  
            if name[-4:]=='.jpg':
                all_files.append(name)  
    return all_files


def load_model(weights_path, device):
    if not os.path.exists(weights_path):
        print("Model weights not found!")
        exit()
    model = YOLO(weights_path).to(device)
    model.fuse()
    model.info(verbose=False)
    return model

def process_images(path, model):
    # if not os.path.exists(path):
    #     print(f"Path {path} does not exist!")
    #     exit()
    images_path=path+'images/test/'
    image_path=path+'image/test/'
    # all_file=list_all_files(images_path)
    all_file=['06144.jpg']
    
    for   i in tqdm(range(len(all_file))):
        files=all_file[i]
        
        pathrgb_ir=[images_path+files,image_path+files]
        imgs=[]
        for img_file in pathrgb_ir:
            if not img_file.endswith(".jpg"):
                continue
            # img_path = os.path.join(path, img_file)
            img = cv2.imread(img_file)
            if img is None:
                print(f"Failed to load image {img_file}")
                continue
            imgs.append(img)
        # 第一个是 rgb ir 第二个是ir
        maskrgb = imgs[0].copy()
        maskir = imgs[1].copy()
        # 第一个rgb 第二个是ir
        imgs= np.concatenate((imgs[0], imgs[1]), axis=2)

            # 定义颜色列表，假设有四个类别  
        colors = [  
            [255, 0, 0],      # 红色，类别0  
            [0, 255, 0],      # 绿色，类别1  
            [0, 0, 255],      # 蓝色，类别2  
            [255, 255, 0],    # 黄色，类别3  
            [75, 0, 130]      # 紫色（深紫色），类别4  
        ]
            
        result = model.predict(imgs,save=True,imgsz=640,visualize=False,obb=True)
        # cls, xywh = result[0].obb.cls, result[0].obb.xywh
        
        cls, xywh = result[0].boxes.cls, result[0].boxes.xywh
        class_conf=result[0].boxes.conf
        class_conf_=class_conf.detach().cpu().numpy()
        cls_, xywh_ = cls.detach().cpu().numpy(), xywh.detach().cpu().numpy()

        for pos, cls_value,cls_conf in zip(xywh_, cls_,class_conf_):
            pt1, pt2 = (np.int_([pos[0] - pos[2] / 2, pos[1] - pos[3] / 2]),
                        np.int_([pos[0] + pos[2] / 2, pos[1] + pos[3] / 2])) 
            color = colors[int(cls_value)]  
            #color = [0, 0, 255] if cls_value == 0 else [0, 255, 0]
            cv2.rectangle(maskrgb, tuple(pt1), tuple(pt2), color, 2)
            cv2.rectangle(maskir, tuple(pt1), tuple(pt2), color, 2)
            # 限制一下标签位置
            xfill=20
            yfill=15
            text_x=pt1[0]
            text_y=pt1[1]
            if(text_x+xfill>img.shape[1]):
                print(text_x)
                text_x=img.shape[1]-30
            if(text_y-yfill<0):
                text_y=pt2[1]+10
            else :
                text_y-=2
            class_names = ["car","truck","bus","van","freight car"]  # 你需要定义这个列表  
            class_name = class_names[int(cls_value)] if int(cls_value) < len(class_names) else "未知类别"  
            # 使用putText添加文本  
            font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型  
            font_scale = 0.5  # 字体大小  
            font_color = color  # 文本颜色  
            thickness = 1  # 线条粗细  
            # 计算文本大小（可选，但有助于定位）  
            text_size = cv2.getTextSize(class_name, font, font_scale, thickness)[0]  
            text_x = max(text_x, 0)  # 确保文本不会超出图像边界  
            text_y = max(text_y, 0)  
            # 在maskrgb上添加文本  
            cv2.putText(maskrgb, class_name+f' {float(cls_conf):.2f}', (text_x, text_y-3), font, font_scale, font_color, thickness)  
            # 如果你也想在maskir上添加文本（通常不需要，但如果需要可以取消注释）  
            cv2.putText(maskir, class_name+f' {float(cls_conf):.2f}', (text_x, text_y-3), font, font_scale, font_color, thickness)  
            cv2.imwrite("./detect/rgb_"+files,maskrgb)
            cv2.imwrite("./detect/ir_"+files,maskir)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if os.path.exists('./detect') !=True:
        os.makedirs('./detect') 

    model = load_model("/home/mjy/ultralytics/pth/baseline.pt", device)
    process_images("/home/mjy/ultralytics/datasets/OBBCrop/", model)

if __name__ == "__main__":
    main()

