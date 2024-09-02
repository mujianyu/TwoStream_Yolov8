# Description: This script is used to crop the images in the dataset.
#  我们需要将DroneVehicle数据集裁剪到640*640
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


dataset_dir_vi = '/home/mjy/ultralytics/datasets/OBB/images/val'
output_dir_vi = '/home/mjy/ultralytics/datasets/OBBCrop/images/val'
dataset_dir_ir = '/home/mjy/ultralytics/datasets/OBB/image/val'
output_dir_ir = '/home/mjy/ultralytics/datasets/OBBCrop/image/val'

# 检查文件夹是否存在，如果不存在则创建
create_file(output_dir_vi, output_dir_ir)
# 获得需要转化的图片路径并生成目标路径
image_filenames_vi = [(os.path.join(dataset_dir_vi, x), os.path.join(output_dir_vi, x))
                      for x in os.listdir(dataset_dir_vi)]

image_filenames_ir = [(os.path.join(dataset_dir_ir, x), os.path.join(output_dir_ir, x))
                      for x in os.listdir(dataset_dir_ir)]
# 转化所有图片
print('Start transforming vision images...')

for path in tqdm(image_filenames_vi):
    update(path[0], path[1])


print('Start transforming infrared images...')
for path in tqdm(image_filenames_ir):
    update(path[0], path[1])
