from ultralytics import YOLO
 


# model = YOLO('/home/mjy/ultralytics/pth/PCONV.engine') 

model = YOLO('/home/mjy/ultralytics/pp/best.engine') 
metrics = model.val(data='/home/mjy/ultralytics/data/drone2.yaml',split='test',imgsz=640,batch=16)
# import torch
# model=torch.load("./baseline.pt")
# # for k,v in type(model['model']):
# #     print(k)
# for  para in model['model'].named_parameters():
#     if(para[1].dtype==torch.float16):
#         print(para[0])
#         print(para[1