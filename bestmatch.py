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


def read_yolo_data(txt_path):
    lines=0
    with open(txt_path, 'r') as file:
        for line in file:
            lines+=1
    return lines


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
    labels_path=path+'labels/test/'
    all_file=list_all_files(images_path)
    
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

        imgs= np.concatenate((imgs[0], imgs[1]), axis=2)


        result = model.predict(imgs,save=True,imgsz=640,visualize=False)
        
        cls, xywh = result[0].boxes.cls, result[0].boxes.xywh
      
        gt=read_yolo_data(labels_path+files[:-4]+'.txt')
        if(len(cls)==gt):
            with open('bestmatch.txt', 'a') as file:  # 使用 'a' 模式以追加方式打开文件  
                # 写入信息到文件  
                file.write("第" + files +"文件:"+str(gt)+"个\n")  # 假设 files 变量已经包含了你想要的文件名或标识符

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_model("/home/mjy/ultralytics/pth/p.pt", device)
    process_images("/home/mjy/ultralytics/datasets/OBBCrop/", model)

if __name__ == "__main__":
    main()

