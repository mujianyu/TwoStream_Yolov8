import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import serial
import time



def load_model(weights_path, device):
    if not os.path.exists(weights_path):
        print("Model weights not found!")
        exit()
    model = YOLO(weights_path).to(device)
    model.fuse()
    model.info(verbose=False)
    return model

def process_images(path, model):
    if not os.path.exists(path):
        print(f"Path {path} does not exist!")
        exit()

    imgs=[]
    for img_file in os.listdir(path):
        if not img_file.endswith(".jpg"):
            continue

        img_path = os.path.join(path, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image {img_path}")
            continue
        
        imgs.append(img)
    cv2.imwrite("result1.jpg",imgs[0])
    cv2.imwrite("result2.jpg",imgs[1])
    # 第一个是 rgb ir 第二个是ir
    mask = imgs[0].copy()
    

    
    # 第一个rgb 第二个是ir
    imgs= np.concatenate((imgs[0], imgs[1]), axis=2)
    cv2.imwrite("result3.jpg",imgs[...,:3])
    cv2.imwrite("result4.jpg",imgs[...,3:])



    result = model.predict(imgs)
    cls, xywh = result[0].boxes.cls, result[0].boxes.xywh
    cls_, xywh_ = cls.detach().cpu().numpy(), xywh.detach().cpu().numpy()

    for pos, cls_value in zip(xywh_, cls_):
        pt1, pt2 = (np.int_([pos[0] - pos[2] / 2, pos[1] - pos[3] / 2]),
                    np.int_([pos[0] + pos[2] / 2, pos[1] + pos[3] / 2]))
        color = [0, 0, 255] if cls_value == 0 else [0, 255, 0]
    cv2.rectangle(mask, tuple(pt1), tuple(pt2), color, 2)

    res_ = "Yes" if np.any(cls_ == 1) else "No"
    print(res_)
    

    cv2.imwrite("result.jpg",mask)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_model("./best.pt", device)
    process_images("./images1/", model)

if __name__ == "__main__":
    main()

