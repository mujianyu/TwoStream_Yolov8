#作者在每张图像的顶部、底部、左侧和右侧设置了一个宽度为100像素的白色边框，因此下载的图像比例为840 x 712，去除周围的白色边框并将图像比例更改为640 x 512。
#由于DroneVehicle数据集中的每张图像的顶部、底部、左侧和右侧设置了一个宽度为100像素的白色边框，因此需要生成去除周围白色边框的图像。
#由于DroneVehicle数据集的标注采用XML格式进行保存，实验需要TXT格式的标注，因此需要对相关数据进行合理转化。


import numpy as np
import cv2
import os
from tqdm import tqdm


def create_file(output_dir_vi, output_dir_ir):
    if not os.path.exists(output_dir_vi):
        os.makedirs(output_dir_vi)
    if not os.path.exists(output_dir_ir):
        os.makedirs(output_dir_ir)
    print(f'Created folder:({output_dir_vi}); ({output_dir_ir})')


def update(input_img_path, output_img_path):
    image = cv2.imread(input_img_path)
    cropped = image[100:612, 100:740]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(output_img_path, cropped)

#dataset_dir_vi：用于存放可见光未处理的图像
#output_dir_vi：用于存放可见光处理后的图像
#dataset_dir_ir：用于存放红外未处理的图像
#output_dir_ir：用于存放红外处理后的图像

dataType=['train','val','test']
for i in range(len(dataType)):
    data=dataType[i]
    dataset_dir_vi = './'+data+'/'+data+'img'
    output_dir_vi = './datasets/images/'+data
    dataset_dir_ir = './'+data+'/'+data+'imgr'
    output_dir_ir = './datasets/image/'+data

    # 检查文件夹是否存在，如果不存在则创建
    create_file(output_dir_vi, output_dir_ir)
    # 获得需要转化的图片路径并生成目标路径
    image_filenames_vi = [(os.path.join(dataset_dir_vi, x), os.path.join(output_dir_vi, x))
                        for x in os.listdir(dataset_dir_vi)]

    image_filenames_ir = [(os.path.join(dataset_dir_ir, x), os.path.join(output_dir_ir, x))
                        for x in os.listdir(dataset_dir_ir)]
    # 转化所有图片
    print('Start transforming vision images '+data)
    for path in tqdm(image_filenames_vi):
        update(path[0], path[1])


    print('Start transforming infrared images '+data)
    for path in tqdm(image_filenames_ir):
        update(path[0], path[1])
