# YOLOV8多模态目标检测

## 1.数据集文件格式
```
xxx
├── image
│   ├── test
│   ├── train
│   └── val
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val
```
images 保存的是rgb图片
image 保存的是ir图片
labels 公用一个标签(一般来说使用ir图片标签)
## 2. 权重文件下载
<details open><summary>目标检测权重</summary>

| 模型                                                                                   | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------- | -------------------- | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640             | 37.3                 | 80.4                        | 0.99                             | 3.2            | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640             | 44.9                 | 128.4                       | 1.20                             | 11.2           | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640             | 50.2                 | 234.7                       | 1.83                             | 25.9           | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640             | 52.9                 | 375.2                       | 2.39                             | 43.7           | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640             | 53.9                 | 479.1                       | 3.53                             | 68.2           | 257.8             |

</details>

<details open><summary>旋转框检测权重</summary>

| 模型                                                                                           | 尺寸<br><sup>(像素) | mAP<sup>test<br>50 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------- | ------------------ | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n-obb](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-obb.pt) | 1024            | 78.0               | 204.77                      | 3.57                             | 3.1            | 23.3              |
| [YOLOv8s-obb](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-obb.pt) | 1024            | 79.5               | 424.88                      | 4.07                             | 11.4           | 76.3              |
| [YOLOv8m-obb](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-obb.pt) | 1024            | 80.5               | 763.48                      | 7.61                             | 26.4           | 208.6             |
| [YOLOv8l-obb](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-obb.pt) | 1024            | 80.7               | 1278.42                     | 11.83                            | 44.5           | 433.8             |
| [YOLOv8x-obb](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-obb.pt) | 1024            | 81.36              | 1759.10                     | 13.23                            | 69.5           | 676.7             |
</details>

## 3. 配置模型yaml文件和数据集yaml文件
分别在yaml文件夹和data文件下进行模型和数据集文件的配置
## 4. obb文件夹
旋转框训练、测试、推理、热图绘制、onnx推理
## 5. hbb文件夹
水平框训练、推理、热图绘制
## 6. exportonnx文件夹
导出onnx格式文件、判断onnx格式和pth格式的输出是否相同
## 7. plot 
绘制hbb真实框