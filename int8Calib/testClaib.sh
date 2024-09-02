# 获取校准集
count=1  
while [ $count -lt 20 ]  
do  
   echo "第${count}次"
   /home/mjy/miniconda3/envs/t2/bin/python int8Claib.py.py
   mv /home/mjy/ultralytics/datasets/Calib2 /home/mjy/ultralytics/datasets/C${count}
   mv /home/mjy/ultralytics/pp/best.engine /home/mjy/ultralytics/pp/b${count}.engine
   mv /home/mjy/ultralytics/pp/best.cache /home/mjy/ultralytics/pp/b${count}.cache	
   count=$((count + 1))
done
 
