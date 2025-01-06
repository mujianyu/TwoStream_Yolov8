import os
import xml.etree.ElementTree as ET
import math
import cv2 as cv
import argparse
from tqdm import tqdm

# 图像类别
classes = ["feright_car", "car", "truck", "bus", "van"]


# 定义相关地址参数
def parse_args():
    parser = parser = argparse.ArgumentParser(description='polygon')


    parser.add_argument('--in_xml_vi_dir', default='./val/vallabel', help='可见光 XML 文件地址')
    parser.add_argument('--in_xml_ir_dir', default='./val/vallabelr', help='红外光 XML 文件地址')
    parser.add_argument('--out_vi_txt_dir', default='./datasets/labelrgb/val', help='可见光 TXT 文件地址')
    parser.add_argument('--out_ir_txt_dir', default='./datasets/labelir/val', help='红外光 TXT 文件地址')
    parser.add_argument('--in_vi_img_dir', default='./datasets/images/val', help='裁切后的 vi-img 地址')
    parser.add_argument('--in_ir_img_dir', default='./datasets/image/val', help='裁切后的 ir-img 地址')

    args = parser.parse_args()
    return args


# 根据 xml 文件中的 name 选择生成的 txt 文件中的 id
def select_id(name):
    if name == "car":
        id = 0
    elif name == "truck":
        id = 1
    elif name == "bus":
        id = 2
    elif name == "van":
        id = 3
    elif name == "feright car":
        id = 4
    else:
        #其它类
        return -1    
    return id


# YOLO 数据处理
def data_transform(height, width, xmin, ymin, xmax, ymax):
    # 中心点坐标 x_c,y_c
    x_c = (xmin + xmax) / 2
    y_c = (ymin + ymax) / 2

    # 中心横坐标与图像宽度比值 x_，中心纵坐标与图像高度比值 y_，bbox 宽度与图像宽度比值 w_，bbox 高度与图像高度比值 h_
    x_ = x_c / width
    y_ = y_c / height
    w_ = (xmax - xmin) / width
    h_ = (ymax - ymin) / height
    
    #归一化
    x_=min(max(0,x_),1)
    w_=min(max(0,w_),1)
    y_=min(max(0,y_),1)
    h_=min(max(0,h_),1)

    return x_, y_, w_, h_


# xml 文件转 txt 文件
def xml2txt(in_xml_dir, xml_name, out_txt_dir, in_img_dir):
    txt_name = xml_name[:-4] + '.txt'   # 获取生成的 txt 文件名
    txt_path = out_txt_dir  # 获取生成的 txt 文件保存地址

    # 判断保存 txt 文件的文件夹是否存在，如果不存在则创建相应文件夹
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    txt_file = os.path.join(txt_path, txt_name)     # 获取 txt 文件地址（保存地址 + 保存名字）

    img_name = xml_name[:-4] + '.jpg'   # 获取图像名字，确保生成的 txt 文件名与图像文件名一致
    img_path = os.path.join(in_img_dir, img_name)   # 获取图像地址
    img = cv.imread(img_path)   # 读取图像信息
    height, width, _ = img.shape    # 获取图像高度（height），宽度（width），通道数（_）

    xml_file = os.path.join(in_xml_dir, xml_name)   # 获取 xml 文件地址
    tree = ET.parse(os.path.join(xml_file))     # 使用 ET.parse 方法解析 xml 文件
    root = tree.getroot()   # 使用 getroot 方法获取根目录

    # 生成对应的 txt 文件
    with open(txt_file, "w+", encoding='UTF-8') as out_file:
        for obj in root.findall('object'):
            # 修改部分标注文件中标注不全的 name 文件
            name = obj.find('name').text
            if name == 'feright_car':
                name = 'feright car'
            else:
                name = name

            # 从 xml 文件中提取相关数据信息,并进行删除白边数据操作（白边宽度 100 像素）
            if obj.find('polygon'):
                # 创建空列表用于存放需要处理的数据
                xmin, xmax, ymin, ymax = [], [], [], []
                polygon = obj.find('polygon')
                # 使用 .find() 方法获取对应 xml 文件中键的键值
                x1 = int(polygon.find('x1').text) - 100
                y1 = int(polygon.find('y1').text) - 100
                x2 = int(polygon.find('x2').text) - 100
                y2 = int(polygon.find('y2').text) - 100
                x3 = int(polygon.find('x3').text) - 100
                y3 = int(polygon.find('y3').text) - 100
                x4 = int(polygon.find('x4').text) - 100
                y4 = int(polygon.find('y4').text) - 100
                # 将获取后的数据填入空列表中
                for i in [x1, x2, x3, x4]:
                    xmin.append(i)
                    xmax.append(i)
                for j in [y1, y2, y3, y4]:
                    ymin.append(j)
                    ymax.append(j)
                # 使用 min()、max() 方法获取最大值，最小值
                xmin = min(xmin)
                xmax = max(xmax)
                ymin = min(ymin)
                ymax = max(ymax)
                # yolo 格式转换
                result = data_transform(height, width, xmin, ymin, xmax, ymax)
                # id 选择
                result_id = select_id(name)
                #标签不属于这五类则返回-1，不继续执行
                if result_id==-1:
                    continue

            elif obj.find('bndbox'):
                bndbox = obj.find('bndbox')
                # 使用 .find() 方法获取对应 xml 文件中键的键值
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text
                x1 = int(xmin) - 100
                y1 = int(ymin) - 100
                x3 = int(xmax) - 100
                y3 = int(ymax) - 100
               
                # yolo 格式转换
                result = data_transform(height, width, x1, y1, x3, y3)
                # id 选择
                result_id = select_id(name)
                #标签不属于这五类则返回-1，不继续执行
                if result_id==-1:
                    continue

            #没有result会报错
            # 创建 txt 文件中的数据
            data = str(result[0]) + " " + str(result[1]) + " " + str(result[2]) + " " + str(result[3]) + '\n'
            data = str(result_id) + " " + data
            out_file.write(data)


if __name__ == "__main__":
    args = parse_args()     # 获取命令参数
    xml_vi_path = args.in_xml_vi_dir    # 获取可见光 xml 文件地址
    xmlFiles_vi = os.listdir(xml_vi_path)   # 生成可见光 xml 文件名列表
    xml_ir_path = args.in_xml_ir_dir    # 获取红外 xml 文件地址
    xmlFiles_ir = os.listdir(xml_ir_path)   # 生成红外 xml 文件名列表

    print('Start transforming vision labels...')
    for i in tqdm(range(0, len(xmlFiles_vi))):
        xml2txt(args.in_xml_vi_dir, xmlFiles_vi[i], args.out_vi_txt_dir, args.in_vi_img_dir)
    print('Finish.')

    print('Start transforming infrared labels...')
    for i in tqdm(range(0, len(xmlFiles_ir))):
        xml2txt(args.in_xml_ir_dir, xmlFiles_ir[i], args.out_ir_txt_dir, args.in_ir_img_dir)
    print('Finish.')