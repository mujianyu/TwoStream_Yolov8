import numpy as np
import tensorrt as trt
import torch
import logging
 
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
 
 
def build_engine(onnx, dynamic=True, half=True):
    # f = onnx.with_suffix('.engine')
    f = 'trt.engine'
    # 1、创建日志记录器
    log = trt.Logger()
    # 2、创建builder对象
    builder = trt.Builder(log)
    # 3、创建 Builder Config 对象
    config = builder.create_builder_config()
    # 4、将workspace*1 二进制左移30位后的10进制
    workspace = 1
    config.max_workspace_size = workspace * 1 << 30         # 设置 TensorRT 推理引擎使用的最大工作空间大小，单位为字节
    # 5、定义networko并加载ONNX解析器
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, log)
 
    if not parser.parse_from_file(str(onnx)):       # 查看是否解析成功
        raise RuntimeError(f'failed to load ONNX file: {onnx}')
 
    # 6、获得网络的输入输出
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
 
    # 7.判断是否动态输入
    if dynamic:
        im = torch.zeros(1,6,640,640)
        if im.shape[0] <= 1:
            # log.warning(f"{trt} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
            print('x')
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)
    # 判断是否支持FP16推理
 
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    # build engine 文件的写入  这里的f是前面定义的engine文件
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        # 序列化model
        t.write(engine.serialize())
    return f, None
 
if __name__ == '__main__':
    engine, context = build_engine(r'./baseline.onnx')