# 测试onnx与torch的输出 
#使用随机输入input1进行测试,看torch下的模型和onnx模型输出是否一致.判断方法为采用np.testing.assert_almost_equal进行测试,判断输出的小数点后三位,如果一致,输出结果为None.
import onnxruntime
import torch
import cv2
import torch
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend

from ultralytics import YOLO
import os 
# def preprocess_warpAffine(image, dst_width=640, dst_height=640):
#     scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
#     ox = (dst_width  - scale * image.shape[1]) / 2
#     oy = (dst_height - scale * image.shape[0]) / 2
#     M = np.array([
#         [scale, 0, ox],
#         [0, scale, oy]
#     ], dtype=np.float32)
#     img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
#                              borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    
#     IM = cv2.invertAffineTransform(M)

#     img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
#     img_pre = img_pre.transpose(2, 0, 1)[None]
#     img_pre = torch.from_numpy(img_pre)
#     return img_pre, IM

# def load_model(weights_path, device):
#     if not os.path.exists(weights_path):
#         print("Model weights not found!")
#         exit()
#     model = YOLO(weights_path).to(device)
#     model.fuse()
#     model.info(verbose=False)
#     return model


import numpy as np
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

onnx_weights = '/home/mjy/ultralytics/runs/detect/train26/weights/best.onnx'  # onnx权重路径
torch_weights = '/home/mjy/ultralytics/runs/detect/train26/weights/best.pt'  # torch权重路径
 
input1 = torch.randn(1, 6, 640, 640, dtype=torch.float32) 

# img = cv2.imread("/home/mjy/ultralytics/datasets/OBBCrop/images/train/00011.jpg")
# imgr=img
# irimg = cv2.imread("/home/mjy/ultralytics/datasets/OBBCrop/image/train/00011.jpg")
# img_pre = preprocess_letterbox(img)
# input1=np.concatenate((img,irimg),axis=2)
# input1, IM = preprocess_warpAffine(input1)


model  = AutoBackend(weights=torch_weights,fp16=False)
names  = model.names
result = model(input1)[0].transpose(-1, -2) 




# model = YOLO(model=torch_weights).to(device)

with torch.no_grad():
    torch_output = model(input1)[0]

img = input1.to('cpu').numpy().astype(np.float32) # array

model.eval()

# 运行模型  
#output = session.run([output_name], {input_name: img})
#print(input_name)
#print(output_name)

session = onnxruntime.InferenceSession(onnx_weights)
input_name = session.get_inputs()[0].name  
output_name = session.get_outputs()[0].name  
  
onnx_output = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))


#判断输出结果是否一致，小数点后3位一致即可
print("onnx的输出")
print(onnx_output[0])
print("torch的输出")
print(torch_output)
#测试输出不同


print("onnx的输出与torch输出是否有不同(精确到小数点后3位)：")
print(np.testing.assert_almost_equal(to_numpy(torch_output), onnx_output[0], decimal=3))

# provider = "CPUExecutionProvider"
# onnx_session = onnxruntime.InferenceSession(onnx_weights, providers=[provider])

# print("----------------- 输入部分 -----------------")
# input_tensors = onnx_session.get_inputs()  # 该 API 会返回列表
# for input_tensor in input_tensors:         # 因为可能有多个输入，所以为列表
    
#     input_info = {
#         "name" : input_tensor.name,
#         "type" : input_tensor.type,
#         "shape": input_tensor.shape,
#     }
#     print(input_info)

# print("----------------- 输出部分 -----------------")
# output_tensors = onnx_session.get_outputs() # 该 API 会返回列表
# for output_tensor in output_tensors:         # 因为可能有多个输出，所以为列表
    
#     output_info = {
#         "name" : output_tensor.name,
#         "type" : output_tensor.type,
#         "shape": output_tensor.shape,
#     }
#     print(output_info)


# print(torch_output.shape)