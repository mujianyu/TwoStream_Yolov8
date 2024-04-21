import shutil

def copy_files_from_list(file_list_path, destination_folder1,destination_folder2):
    # 打开文件列表
    with open(file_list_path, 'r') as file:
        # 逐行读取文件路径
        for line in file:
            # 去除每行末尾的换行符并获取文件路径
            file_path = line.strip()

            # 复制文件到目标文件夹
            shutil.copy(file_path, destination_folder1)
            # file_path=file_path
            # # 将 "images" 替换为 "labels"
            # new_file_path = file_path.replace("images", "labels")

            # # 将 ".jpg" 替换为 ".txt"
            # new_file_path = new_file_path.replace(".jpg", ".txt")
            # shutil.copy(new_file_path, destination_folder2)

# 指定文件列表路径和目标文件夹
file_list_path = '/home/mjy/ultralytics/datasets/datasets/IR/val.txt'  # 假设文件列表保存为file_list.txt
destination_folder1 = '/home/mjy/ultralytics/datasets/rgbir/image/val'  # 指定目标文件夹路径 image
destination_folder2 = '/home/mjy/ultralytics/datasets/rgbir/labels/val'  # 指定目标文件夹路径 label
# 调用函数复制文件
copy_files_from_list(file_list_path, destination_folder1,destination_folder2)
