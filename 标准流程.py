# 以CBAM为例子

# 需要装载drive

#------------------------------命名---------------------------------#
method="CBAM"
baseline_floader="baseline"
method_floader="meta-baseline-%s"%method
root="/content/%s/%s"%(baseline_floader,method_floader)
drive_data_path="/content/drive/MyDrive/few_shot_meta_baseline/materials/miniImageNet.zip"
drive_baseline_floader="Meta-baseline"
drive_method_floader="meta-baseline-%s"%method


#############  准备阶段  ############# 
# 下载代码 # 
%cd /content
!echo "下载代码"
!git clone https://github.com/Jf-Chen/baseline.git
# 从云盘装载数据集 #
data_path="%s/materials"%(root)
!echo "从google drive装载"
%cd $data_path
!cp $drive_data_path $data_path
!unzip miniImageNet.zip -d ./mini-imagenet
!echo "data_path/mini-ImageNet 装载、解压完成"
#---------------end------------------------
!echo "准备环境"
!pip3 install tensorboardX
!pip3 install -U PyYAML
!echo "已安装"

#############  train classifer  ############# 


from datetime import datetime
now = datetime.now() # current date and time
current_time= now.strftime("%YY-%mM-%dD-%HH-%MMinu")
print("current_time==",current_time)

%cd $root
!chmod +777 train_meta.py
!python train_meta.py --config configs/train_meta_mini.yaml

%cd $root
#------------保存classifer到云盘------------#
stage="train-classifer"
!tar -czf save.tar.gz ./save
drive_save_path= "/content/drive/MyDrive/%s/%s/%s/%s" %(drive_baseline_floader,drive_method_floader,stage,current_time)
print("train_classifer_mini save 保存的位置:",drive_save_path)
!mkdir -p drive_save_path
!cp save.tar.gz $drive_save_path
!cp $root/configs/train_classifer_mini.yaml $drive_save_path


#############  元训练阶段  ############# 

from datetime import datetime
now = datetime.now() # current date and time
current_time= now.strftime("%YY-%mM-%dD-%HH-%MMinu")
print("current_time==",current_time)

%cd $root
!chmod +777 train_meta.py
!python train_meta.py --config configs/train_meta_mini.yaml

%cd $root
!tar -czf save.tar.gz ./save
drive_save_path= "/content/drive/MyDrive/%s/%s/%s/%s" %(drive_baseline_floader,drive_method_floader,stage,current_time)
print("train_meta_mini  save 保存的位置:",drive_save_path)
!mkdir -p drive_save_path
!cp save.tar.gz $drive_save_path
!cp $root/configs/train_meta_mini.yaml $drive_save_path


#############  测试阶段  ############# 


