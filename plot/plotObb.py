# Description: 读取图像和对应的txt文件，绘制有向边框
import cv2  
import os  
import numpy as np  
  
def draw_quadrilateral(image_path, box_file_path):  
    # 读取图像  
    image = cv2.imread(image_path)  
    if image is None:  
        print(f"Error: Could not read image at {image_path}")  
        return  
  
    # 读取txt文件中的边框坐标  
    with open(box_file_path, 'r') as f:  
        boxes = f.read().strip().split('\n')  # 使用空格或逗号作为分隔符  
    
            # 遍历每个四边形坐标集  
        for box in boxes: 
           
            coords = box.split()  # 使用空格作为分隔符  
            coords=coords[1:]
            for i in range(len(coords)):
                if i %2==0:
                    coords[i]=float(coords[i])*640
                else :
                    coords[i]=float(coords[i])*512
            if len(coords) != 8:  
                print(f"Warning: Invalid number of coordinates for a box in {box_file_path}")  
                continue  
    
            # 将字符串坐标转换为整数并分组为四个点  
            pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))  
            first_pt = pts[0, 0, :]  # 获取第一个点的坐标  
            cv2.circle(image, tuple(first_pt), 3, (0, 0, 255), -1)  # -1表示填充整个圆
            # 绘制四边形边框（这里使用绿色，线宽为2）  
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)  
    
        # 显示或保存图像（这里选择保存）  
    base_name = os.path.splitext(os.path.basename(image_path))[0]  

    output_path = os.path.join('/home/mjy/ultralytics/datasets/test', base_name + '_with_box.jpg') 
    cv2.imwrite(output_path, image)  
    print(f"Saved image with boxes to {output_path}")  
    

  
# 示例：遍历图像目录并处理每个图像和对应的txt文件  
image_dir = '/home/mjy/ultralytics/datasets/OBBCrop/images/val'  # 替换为你的图像目录  
label_dir= '/home/mjy/ultralytics/datasets/OBBCrop/labels/val' 
for filename in os.listdir(image_dir):  
    if filename.endswith('.jpg') or filename.endswith('.jpg'):  # 假设你的图像是jpg或jpg格式  
        image_path = os.path.join(image_dir, filename)  
        box_file_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')  # 假设txt文件和图像文件同名，只是扩展名不同  
        
        if os.path.exists(box_file_path):  
            draw_quadrilateral(image_path, box_file_path)  
        else:  
            print(f"Warning: No corresponding txt file found for {image_path}")