# 根据源数据集生成校准集 根据校准集数据进行int8量化
import os  
import shutil  
import random  
  
def copy_matching_files(src_images_dir, src_labels_dir,src_image_dir, dest_images_dir, dest_labels_dir, dest_image_dir,num_files):  
    """  
    从src_images_dir和src_labels_dir中挑选文件名相匹配的.jpg和.txt文件，  
    并复制num_files个这样的文件对到dest_images_dir和dest_labels_dir。  
    """  
    if not os.path.exists(dest_images_dir):  
        os.makedirs(dest_images_dir)  
    if not os.path.exists(dest_labels_dir):  
        os.makedirs(dest_labels_dir)  
    if not os.path.exists(dest_image_dir):  
        os.makedirs(dest_image_dir)  
  
    # 获取所有.jpg文件名（不带扩展名）  
    image_files = {os.path.splitext(f)[0] for f in os.listdir(src_images_dir) if f.endswith('.jpg')}  
    # 获取所有.txt文件名（不带扩展名），并筛选出与.jpg文件名相匹配的  
    label_files = {os.path.splitext(f)[0] for f in os.listdir(src_labels_dir) if f.endswith('.txt') and os.path.splitext(f)[0] in image_files}  
  
    # 确保有足够的匹配文件  
    if len(label_files) < num_files:  
        raise ValueError(f"Not enough matching files ({len(label_files)}) to copy {num_files} pairs.")  
    # 随机挑选文件（这里我们实际上是在挑选文件名，因为文件名已经匹配了）  
    selected_files = random.sample(list(label_files), num_files)  
    # 复制文件  
    for filename in selected_files:  
        img_src_path = os.path.join(src_images_dir, f"{filename}.jpg")  
        irimg_src_path = os.path.join(src_image_dir, f"{filename}.jpg")  
        label_src_path = os.path.join(src_labels_dir, f"{filename}.txt")  
        img_dest_path = os.path.join(dest_images_dir, f"{filename}.jpg")  
        irimg_dest_path = os.path.join(dest_image_dir, f"{filename}.jpg")  
        label_dest_path = os.path.join(dest_labels_dir, f"{filename}.txt")  

        shutil.copy(img_src_path, img_dest_path)  
        shutil.copy(label_src_path, label_dest_path)  
        shutil.copy(irimg_src_path, irimg_dest_path)  
  
# 设置源目录和目标目录  
src_images = './datasets/OBBCrop/images/train'  
src_image = './datasets/OBBCrop/image/train'  
src_labels = './datasets/OBBCrop/labels/train'  
# 假设你实际上没有image目录，或者它有与images/train不同的用途，这里不处理它  
# 但如果你确实需要处理它（且它与labels相匹配），请取消下面几行的注释并相应地调整  
# src_image = 'OBBCrop/image/train'  
# ... 并在函数中添加对它的处理逻辑（但请注意，这通常是不必要的）  
dest_base = './datasets/Calib2'  
dest_images = os.path.join(dest_base, 'images/train')  
dest_labels = os.path.join(dest_base, 'labels/train')  
dest_image = os.path.join(dest_base, 'image/train')  
num_files_to_copy=1792
# 复制匹配的文件  
copy_matching_files(src_images, src_labels, src_image,dest_images, dest_labels,dest_image, num_files_to_copy) 

# 生成cache
from ultralytics import YOLO
import ultralytics.nn.tasks
model = YOLO('./yaml/Fasteryolov8n.yaml')
# Train the mod
results = model.train(data='/home/mjy/ultralytics/data/drone3.yaml',batch=16,epochs=1)

# 根据校准集生成int8 engine 
model = YOLO("/home/mjy/ultralytics/pp/best.pt")
model.export(format='engine',int8=True,dynamic=True,batch=16,data='/home/mjy/ultralytics/data/drone3.yaml')

# 测试
model = YOLO('/home/mjy/ultralytics/pp/best.engine') 
metrics = model.val(data='/home/mjy/ultralytics/data/drone2.yaml',split='test',imgsz=640,batch=16)