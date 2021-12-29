#===================下载代码、环境=======================#
%cd /content
!echo "下载代码"
!git clone https://github.com/Jf-Chen/baseline.git
!pip3 install tensorboardX
!pip3 install -U PyYAML

#=================下载数据集=========================#
# 数据集
miniImageNet_drive_path="/content/drive/MyDrive/few_shot_meta_baseline/materials/miniImageNet.zip"
miniImageNet_save_path="%s/miniImageNet.zip"%(root)
miniImageNet_use_path="%s/materials/mini-imagenet"%(root)
miniImageNet_format="zip"

os.chdir(root)
shutil.copy(miniImageNet_drive_path, root)
shutil.unpack_archive(miniImageNet_save_path, miniImageNet_use_path, miniImageNet_format)
# 先创建save
save_path=os.path.join(root,"save")
os.makedirs(save_path,exist_ok=True)

#=================下载预训练文件=========================#
# 预训练文件
drive_root="/content/drive/MyDrive"
last_time=""
stage=""
tarfile_name=""
# classifier_drive_path=os.path.join(drive_root,baseline,method,stage,last_time,tarfile_name)
#classifier_name="classifier_mini_5_way_5_shot_mini.tar.gz"
#classifier_drive_path = "/content/drive/MyDrive/meta-baseline/fcanet-inMeta-inClass-cutmix/classifier/2021Y-12M-16D-05H-57Minu/classifier_mini_5_way_5_shot_mini.tar.gz"
#classifier_dir_name="classifier_mini_5_way_5_shot_mini"

classifier_drive_path = "/content/drive/MyDrive/Meta-baseline/meta-baseline-fcanet-inMeta-inClass/train-classifer/2021Y-12M-01D-07H-00Minu/save.tar.gz"
shutil.copy(classifier_drive_path, root)
shutil.unpack_archive(classifier_drive_path, root, format='gztar')