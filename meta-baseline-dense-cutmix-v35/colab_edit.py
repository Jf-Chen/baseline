#======================文件结构====================#
# github:
# --repository
#  |--baseline-method
#
# google drive
# --baseline
#  |--method
#     |--stage
#        |--time
#           |--file  # save.tar.gz
#
# file中只保存./save下一级的内容，而不是打包整个./save

####
####
####

#======================命名====================#
# 代码
method="fcanet-inMeta-inClass-bsnet-weight-v3"
repository="baseline"
baseline="meta-baseline"
root="/content/%s/%s-%s"%(repository,baseline,method)

# 数据集
miniImageNet_drive_path="/content/drive/MyDrive/few_shot_meta_baseline/materials/miniImageNet.zip"
miniImageNet_save_path="%s/miniImageNet.zip"%(root)
miniImageNet_use_path="%s/materials/mini-imagenet"%(root)
miniImageNet_format="zip"

# 预训练文件
drive_root="/content/drive/MyDrive"
last_time=""
stage="classifier"
tarfile_name="classifier.tar.gz"
# classifier_drive_path=os.path.join(drive_root,baseline,method,stage,last_time,tarfile_name)
classifier_name="save.tar.gz"
classifier_drive_path = "/content/drive/MyDrive/Meta-baseline/meta-baseline-fcanet-inMeta-inClass/train-classifer/2021Y-12M-01D-07H-00Minu/save.tar.gz"

import os
import shutil 
import tarfile

####
####
####

#===================下载代码、环境=======================#
%cd /content
!echo "下载代码"
!git clone https://github.com/Jf-Chen/baseline.git
!pip3 install tensorboardX
!pip3 install -U PyYAML

#=================下载数据集=========================#


os.chdir(root)
shutil.copy(miniImageNet_drive_path, root)
shutil.unpack_archive(miniImageNet_save_path, miniImageNet_use_path, miniImageNet_format)
# 先创建save
save_path=os.path.join(root,"save")
os.makedirs(save_path,exist_ok=True)

#=================下载预训练文件=========================#
print("classifier阶段pth在云盘中的位置：",classifier_drive_path)
shutil.copy(classifier_drive_path, root)
# 解压
# shutil.unpack_archive(os.path.join(root,classifier_name), save_path, 'gztar')
shutil.unpack_archive(os.path.join(root,classifier_name), root, 'gztar') # 因为是旧版本的classifier，不是本文格式生成的
os.chdir(root)


####
####
####

#=================训练meta=========================#
stage="meta"
dataset="mini"
tag="5_way_5_shot_mini"
filename= "train_%s.py"%(stage)
yamlfilename = "configs/train_%s_%s.yaml"%(stage,dataset)

os.chdir(root)
os.chmod(filename,0o777)
# !chmod +777 train_classifier.py
!python $filename --config $yamlfilename

#=================保存classifier/meta.tar.gz到云盘=========================#
from datetime import datetime
now = datetime.now() # current date and time
current_time = now.strftime("%YY-%mM-%dD-%HH-%MMinu")
print("current_time==",current_time)
os.chdir(root)

#=================用epoch-last测试meta=========================#
tag="5_way_5_shot_mini"
filename= "test_few_shot.py"
yamlfilename = "configs/test_few_shot.yaml"

os.chdir(root)
os.chmod(filename,0o777)
!python $filename --shot 5

os.chdir(root)
os.chmod(filename,0o777)
!python $filename --shot 1

# 压缩
# /content/baseline/meta-baseline-fcanet-inMeta-inClass-bsnet-weight/save/meta_mini-imagenet-5shot_meta-baseline-att-resnet12-wide-att
os.chdir(root)
stage_dir = "meta_mini-imagenet-5shot_meta-baseline-att-resnet12-wide-att"
stage_dir_in_root=os.path.join(root,"save",stage_dir ) # 手动输入完整路径,需要获取classifier的文件夹名称，path可根据config命名得知
tar_name = stage+"_"+dataset+"_"+tag
shutil.make_archive(base_name=tar_name, format="gztar", root_dir=stage_dir_in_root) #root/.tar.gz
tar_gz_name =tar_name + ".tar.gz"

# 创建目录
current_stage_dir_drive=os.path.join(drive_root,baseline,method,stage,current_time)
os.makedirs(current_stage_dir_drive,exist_ok=True)

# 拷贝到drive
shutil.copy("%s"%(tar_gz_name),current_stage_dir_drive)
shutil.copy(os.path.join(root,'configs','train_%s_%s.yaml'%(stage,dataset)),
                        current_stage_dir_drive)
shutil.copy(os.path.join(root,'train_meta.py'),
                        current_stage_dir_drive)
print("%s 保存位置："%tar_gz_name, current_stage_dir_drive)
# 注意是小写的meta-baseline
