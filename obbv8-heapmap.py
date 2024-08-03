import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))
            

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    # 修改后处理
    def post_process(self, result):
        logits_ = result[:, 4:-1]
        boxes_ = result[:, :4]
        rotate =result[:,-1]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)

        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(rotate, dim0=0, dim1=1)[indices[0]],xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
  
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, rotate,post_boxes = self.post_process(model_output[0])

        return [[post_result, pre_post_boxes,rotate]]

    def release(self):
        for handle in self.handles:
            handle.remove()

def xywhr2xyxyxyxy(center):
    # reference: https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/utils/ops.py#L545
    is_numpy = isinstance(center, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)

    ctr = center[..., :2]
    w, h, angle = (center[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    #return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)
    return torch.cat([pt1[:,0].unsqueeze(1),pt1[:,1].unsqueeze(1) ,pt2[:,0].unsqueeze(1),pt2[:,1].unsqueeze(1), pt3[:,0].unsqueeze(1),pt3[:,1].unsqueeze(1), pt4[:,0].unsqueeze(1),pt4[:,1].unsqueeze(1)],axis=1) 


class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
    
    def forward(self, data):
        post_result, pre_post_boxes,rotate = data
        xywhr=torch.cat((pre_post_boxes,rotate),axis=1)
        # xy=xywhr2xyxyxyxy(xywhr)
        
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(5):
                    #result.append(pre_post_boxes[i, j])
                    result.append(xywhr[i,j])
        return sum(result)

class yolov8_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        model = attempt_load_weights(weight, device)
        model.info()

        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        
        target = yolov8_target(backward_type, conf_threshold, ratio)
       
        target_layers = [model.model[l] for l in layer]

    

        method = eval(method)(model, target_layers, use_cuda=device.type == 'cuda')
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
        
        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int64)
        self.__dict__.update(locals())
    
    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65,rotated=True,heat=True)[0]
        return result

    def draw_detections(self, box, color, name, img):

        xywh=box[:4]
        r=box[-1]
        xywh=np.expand_dims(xywh,axis=0)

        r=np.expand_dims(r,axis=0)
        r=np.expand_dims(r, axis=1) 
        
        xywhr=np.concatenate((xywh,r),axis=-1)
        xywhr=torch.tensor(xywhr)

        xy=xywhr2xyxyxyxy(xywhr)
        for x1, y1, x2, y2,x3,y3,x4,y4 in xy:
            xmin=min(min(min(x1,x2),x3),x4)
            xmax=max(max(max(x1,x2),x3),x4)

            ymin=min(min(min(y1,y2),y3),y4)
            ymax=max(max(max(y1,y2),y3),y4)
            pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)  
  
            # 定义颜色（BGR）  
            color = (0, 0, 255)# 替换为具体的BGR值  
            
            # 使用cv2.line函数绘制四边形的四条边  
            cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, thickness=2)  
            cv2.line(img, tuple(pts[1]), tuple(pts[2]), color, thickness=2)  
            cv2.line(img, tuple(pts[2]), tuple(pts[3]), color, thickness=2)  
            cv2.line(img, tuple(pts[3]), tuple(pts[0]), color, thickness=2)

        xmin=int(torch.round(xmin))
        xmax=int(torch.round(xmax))
        ymin=int(torch.round(ymin))
        ymax=int(torch.round(ymax))
        text = str(name)  
        position = (xmin-3, ymin - 5) # 文本位置，左上角为原点  
        fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型  
        fontScale = 0.4  # 字体缩放系数  
        color = (0, 0, 255)  # 字体颜色，白色  
        thickness = 1  # 字体粗细，这里设置为3  
        lineType = cv2.LINE_AA  # 线条类型，使用抗锯齿线条  

        #xmin, ymin, xmax, ymax = list(map(int, list(box)))
        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, text, position, fontFace, fontScale, color, thickness,lineType)
        # cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        xywh=boxes[:,:4]
        r=boxes[:,-1]
        r=np.expand_dims(r, axis=1) 
        
        xywhr=np.concatenate((xywh,r),axis=-1)
        xywhr=torch.tensor(xywhr)
        xy=xywhr2xyxyxyxy(xywhr)
        for x1, y1, x2, y2,x3,y3,x4,y4 in xy:
            xmin=min(min(min(x1,x2),x3),x4)
            xmax=max(max(max(x1,x2),x3),x4)

            ymin=min(min(min(y1,y2),y3),y4)
            ymax=max(max(max(y1,y2),y3),y4)

            xmin=int(torch.round(xmin))
            xmax=int(torch.round(xmax))
            ymin=int(torch.round(ymin))
            ymax=int(torch.round(ymax))

            xmin, ymin = max(xmin, 0), max(ymin, 0)
            xmax, ymax = min(grayscale_cam.shape[1] - 1, xmax), min(grayscale_cam.shape[0] - 1, ymax)

            renormalized_cam[ymin:ymax, xmin:xmax] = scale_cam_image(grayscale_cam[ymin:ymax, xmin:xmax].copy())    
        #renormalized_cam = scale_cam_image(grayscale_cam.copy())    
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
    
    def process(self, img_path,imgir_path, save_path):
        # img process
        img = cv2.imread(img_path)
        img_rgb=img
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
       
        
        
        
        # print(img.shape)

        imgir = cv2.imread(imgir_path)
        img_ir=imgir
        imgir = letterbox(imgir)[0]
        imgir = cv2.cvtColor(imgir, cv2.COLOR_BGR2RGB)
        imgir = np.float32(imgir) / 255.0
 

        img=np.concatenate((img,imgir),axis=2)

        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            return
        
        grayscale_cam = grayscale_cam[0, :]
        
        # cam_image = show_cam_on_image(img[...,:3], grayscale_cam, use_rgb=True)
        tensor = torch.tensor(tensor, requires_grad=True) #True

        pred = self.model(tensor)[0]

        pred = self.post_process(pred)

        # rgb 上展示
        img=img[...,3:]
        # if self.renormalize:
        #     cam_image = self.renormalize_cam_in_bounding_boxes(pred.cpu().detach().numpy().astype(np.int32), img, grayscale_cam)
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                # cam_image = self.draw_detections(data, self.colors[int(data[4:-1].argmax())], f'{self.model_names[int(data[4:-1].argmax())]} {float(data[4:-1].max()):.2f}', cam_image)
                cam_image = self.draw_detections(data, self.colors[int(data[4:-1].argmax())], f'{self.model_names[int(data[4:-1].argmax())]} {float(data[4:-1].max()):.2f}',img_rgb)
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)
    
    def __call__(self, img_path, imgir_path,save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            self.process(img_path, imgir_path,f'{save_path}/result.png')
        
def get_params():
    params = {
        'weight': '/home/mjy/ultralytics/pth/p.pt', # 现在只需要指定权重即可,不需要指定cfg
        'device': 'cuda:0',
        'method': 'GradCAM', # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        'layer': [20],
        'backward_type': 'all', # class, box, all
        'conf_threshold': 0.2, # 0.2
        'ratio': 0.02, # 0.02-0.1
        'show_box': True,
        'renormalize': True
    }
    return params

if __name__ == '__main__':
    model = yolov8_heatmap(**get_params())
    # model(r'/home/hjj/Desktop/dataset/dataset_visdrone/VisDrone2019-DET-test-dev/images/9999947_00000_d_0000026.jpg', 'result')
    model(r'/home/mjy/ultralytics/datasets/OBBCrop/images/test/00036.jpg',r'/home/mjy/ultralytics/datasets/OBBCrop/image/test/00036.jpg', 'result')