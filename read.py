import json    
# 原始文件路径  
original_file_path = '/home/mjy/ultralytics/engine/best.engine'  
# 新文件路径（不包含元数据的文件）  
new_file_path = '/home/mjy/ultralytics/engine/no_meta.engine'  
    


with open(original_file_path, "rb") as f:  
    # 读取元数据长度（4字节）  
    meta_len = int.from_bytes(f.read(4), byteorder="little", signed=False)  # 注意：通常长度不会是signed的  
      
    # 跳过元数据部分  
    f.seek(meta_len, 1)  # 使用1作为whence参数，表示从当前位置开始跳过meta_len字节  
      
    # 读取剩余的模型数据  
    model_data = f.read()  
  
# 将模型数据写入新文件  
with open(new_file_path, "wb") as f:  
    f.write(model_data)

print("删除meta")