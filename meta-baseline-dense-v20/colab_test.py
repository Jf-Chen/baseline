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


## **** **** **** **** **** ##
## ****      命名      **** ##
## **** **** **** **** **** ##


#======================命名====================#
# 代码
method="fcanet-inMeta-inClass-cutmix-v2"
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
#classifier_name="classifier_mini_5_way_5_shot_mini.tar.gz"
#classifier_drive_path = "/content/drive/MyDrive/meta-baseline/fcanet-inMeta-inClass-cutmix/classifier/2021Y-12M-16D-05H-57Minu/classifier_mini_5_way_5_shot_mini.tar.gz"
#classifier_dir_name="classifier_mini_5_way_5_shot_mini"

import os
import shutil 
import tarfile


## **** **** **** **** **** ##
## **** 加载预训练文件 **** ##
## **** **** **** **** **** ##


# 预训练文件
drive_root="/content/drive/MyDrive"
last_time="YY-mM-dD-HH-MMinu"
stage="classifier"
tarfile_name="classifier_mini_cutmix.tar.gz"
classifier_drive_path=os.path.join(drive_root,baseline,method,stage,last_time,tarfile_name)

#classifier_drive_path = "/content/drive/MyDrive/meta-baseline/fcanet-inMeta-inClass-cutmix/classifier/2021Y-12M-16D-05H-57Minu/classifier_mini_5_way_5_shot_mini.tar.gz"

#=================下载预训练文件=========================#
# classifier在root中的path
cla_save_path=os.path.join(root,"save_test")
os.makedirs(cla_save_path,exist_ok=True)
print("classifier阶段pth在云盘中的位置：",classifier_drive_path)
shutil.copy(classifier_drive_path, root)
# 解压
# shutil.unpack_archive(os.path.join(root,classifier_name), save_path, 'gztar')
shutil.unpack_archive(os.path.join(root,tarfile_name), cla_save_path, 'gztar') # 因为是旧版本的classifier，不是本文格式生成的
os.chdir(root)


## **** **** **** **** **** ##
## **** 测试classifier **** ##
## **** **** **** **** **** ##


#=================训练classifier=========================#
stage="classifier"
dataset="mini"
tag="cutmix"
# filename= "train_%s.py"%(stage)
filename = "train_classifier_cutmix.py"
yamlfilename = "configs/train_%s_%s_test.yaml"%(stage,dataset)
tar_name = stage+"_"+dataset+"_"+tag 

os.chdir(root)
os.chmod(filename,0o777)
# !chmod +777 train_classifier.py
!python $filename --config $yamlfilename --name $tar_name



#=================保存classifier/meta.tar.gz到云盘=========================#
from datetime import datetime
now = datetime.now() # current date and time
# current_time = now.strftime("%YY-%mM-%dD-%HH-%MMinu")
current_time = "YY-mM-dD-HH-MMinu"
print("current_time==",current_time)
os.chdir(root)


os.chdir(root)
stage_dir = tar_name
tar_root_dir=os.path.join(root,"save" ) # 手动输入完整路径,需要获取classifier的文件夹名称，path可根据config命名得知
tar_base_dir=os.path.join(stage_dir )

# 这里要修改，把文件夹的名字带上
shutil.make_archive(base_name=tar_name, format="gztar", root_dir=tar_root_dir, base_dir=tar_base_dir ) #root/.tar.gz
tar_gz_name =tar_name + ".tar.gz"

# 创建目录
current_stage_dir_drive=os.path.join(drive_root,baseline,method,stage,current_time)
os.makedirs(current_stage_dir_drive,exist_ok=True)


# 拷贝到drive
shutil.copy("%s"%(tar_gz_name),current_stage_dir_drive)
shutil.copy(os.path.join(root,'configs','train_%s_%s_test.yaml'%(stage,dataset)),
                        current_stage_dir_drive)
shutil.copy(os.path.join(root,'train_classifier_cutmix.py'),
                        current_stage_dir_drive)
print("%s 保存位置："%tar_gz_name, current_stage_dir_drive)
# 注意是小写的meta-baseline

# classifier阶段pth在云盘中的位置： /content/drive/MyDrive/meta-baseline/fcanet-inMeta-inClass-cutmix/classifier/2021Y-12M-16D-05H-57Minu/classifier_mini_5_way_5_shot_mini.tar.gz


## **** **** **** **** **** ##
## **** 测试 meta      **** ##
## **** **** **** **** **** ##


#=================训练meta=========================#
stage="meta"
dataset="mini"
tag="5_way_5_shot_mini"
filename= "train_%s.py"%(stage)
yamlfilename = "configs/train_%s_%s_test.yaml"%(stage,dataset)
tar_name = stage+"_"+dataset+"_"+tag

os.chdir(root)
os.chmod(filename,0o777)
# !chmod +777 train_classifier.py
!python $filename --config $yamlfilename  --name $tar_name

#=================保存classifier/meta.tar.gz到云盘=========================#
from datetime import datetime
now = datetime.now() # current date and time
# current_time = now.strftime("%YY-%mM-%dD-%HH-%MMinu")
current_time = "YY-mM-dD-HH-MMinu"
print("current_time==",current_time)
os.chdir(root)


#=================用epoch-last测试meta=========================#
tag="5_way_5_shot_mini"
filename= "test_few_shot.py"
yamlfilename = "configs/test_few_shot.yaml"

os.chdir(root)
os.chmod(filename,0o777)
!python $filename --shot 5 --test-epochs 1

os.chdir(root)
os.chmod(filename,0o777)
!python $filename --shot 1 --test-epochs 1


# 压缩
# /content/baseline/meta-baseline-fcanet-inMeta-inClass-bsnet-weight/save/meta_mini-imagenet-5shot_meta-baseline-att-resnet12-wide-att
os.chdir(root)
stage_dir = tar_name
# /content/baseline/meta-baseline-fcanet-inMeta-inClass-cutmix/save/meta_mini_5_way_5_shot_mini
tar_root_dir=os.path.join(root,"save" ) # 一律按此方式命名
tar_base_dir=os.path.join(stage_dir )
shutil.make_archive(base_name=tar_name, format="gztar", root_dir=tar_root_dir, base_dir=tar_base_dir ) #root/.tar.gz
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


## **** **** **** **** ****  ******** ##
## **** 测试 加载meta训练文件    **** ##
## **** **** **** **** **** **** **** ##


# 预训练文件
drive_root="/content/drive/MyDrive"
last_time="YY-mM-dD-HH-MMinu"
stage="meta"
tarfile_name="meta_mini_5_way_5_shot_mini.tar.gz"
classifier_drive_path=os.path.join(drive_root,baseline,method,stage,last_time,tarfile_name)

#classifier_drive_path = "/content/drive/MyDrive/meta-baseline/fcanet-inMeta-inClass-cutmix/classifier/2021Y-12M-16D-05H-57Minu/classifier_mini_5_way_5_shot_mini.tar.gz"

#=================下载预训练文件=========================#
# classifier在root中的path
cla_save_path=os.path.join(root,"save_test")
os.makedirs(cla_save_path,exist_ok=True)
print("classifier阶段pth在云盘中的位置：",classifier_drive_path)
shutil.copy(classifier_drive_path, root)
# 解压 
# shutil.unpack_archive(os.path.join(root,classifier_name), save_path, 'gztar')
shutil.unpack_archive(os.path.join(root,tarfile_name), cla_save_path, 'gztar') # 因为是旧版本的classifier，不是本文格式生成的
os.chdir(root)





