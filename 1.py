import cv2  
import numpy as np  
import onnxruntime as ort  
import time  
from tqdm import tqdm
  
def estimate_max_batch_size(sess, image_shape, max_memory_gb=10):  
    # 假设每个float32元素占用4字节，计算单张图片所需内存  
    # 注意：这里假设的是6通道，实际可能因数据类型和ONNX Runtime内部优化有所不同  
    # 但这提供了一个大致的估算  
    image_size_bytes = np.prod(image_shape) * 4  # 640*640*6*4  
    max_images_in_memory = int(max_memory_gb * 1024**3 / image_size_bytes)  
    return max_images_in_memory  
  
def test_onnx_model(model_path, image_shape=(6,640, 640), max_batch_size=None):  
    # 加载ONNX模型  
    sess = ort.InferenceSession(model_path,providers= ['CUDAExecutionProvider', 'CPUExecutionProvider'])  
    input_name = sess.get_inputs()[0].name  
  
  
    # 估算最大batch size  
    if max_batch_size is None:  
        max_batch_size = estimate_max_batch_size(sess, image_shape)  
        print(f"Estimated maximum batch size: {max_batch_size}")  
    max_batch_size=1
    # 创建一些随机数据作为输入  
    np.random.seed(0)  
    images = np.random.randn(max_batch_size, *image_shape).astype(np.float32)  
    print(images.shape)
    # 推理前准备
    t=0  
    for i in tqdm(range(100)):
        start_time = time.time()  
       # 重复多次推理以计算平均FPS  
        sess.run(None, {"images": images})  
        end_time = time.time()
        t+=end_time-start_time  
  
    # 计算FPS  
    elapsed_time = t  
    fps = 100 / elapsed_time  
    print(f"FPS: {fps:.2f}")  

    
# 使用示例  
model_path = './baseline.onnx'  
test_onnx_model(model_path)
