## 测试通过后，将yamlfilename, current_time, yamlfilename, --test-epochs修改回来

#===== 1 ============训练classifier=========================#
# yamlfilename = "configs/train_%s_%s_test.yaml"%(stage,dataset)
#===== 2 ============保存classifier/meta.tar.gz到云盘=========================#
# current_time = now.strftime("0Y-0M-0D-0H-0Minu")
#===== 3 ============训练meta=========================#
# yamlfilename = "configs/train_%s_%s_test.yaml"%(stage,dataset)
#===== 4 ============保存meta.tar.gz到云盘=========================#
# current_time = now.strftime("0Y-0M-0D-0H-0Minu")
#===== 5 ===========用epoch-last测试meta=========================#
# !python $filename --shot 5 --test-epochs 1
# !python $filename --shot 1 --test-epochs 1

"""

#=================训练classifier=========================#
stage="classifier"
dataset="mini"
tag="none"
# filename= "train_%s.py"%(stage)
filename = "train_classifier_cutmix.py"
yamlfilename = "configs/train_%s_%s.yaml"%(stage,dataset)  #================================================================== 1 =============================================================#
# yamlfilename = "configs/train_%s_%s_test.yaml"%(stage,dataset)
tar_name = stage+"_"+dataset+"_"+tag 

os.chdir(root)
os.chmod(filename,0o777)
# !chmod +777 train_classifier.py
!python $filename --config $yamlfilename --name $tar_name



#=================保存classifier/meta.tar.gz到云盘=========================#
from datetime import datetime
now = datetime.now() # current date and time
current_time = now.strftime("%YY-%mM-%dD-%HH-%MMinu") #========================================================================= 2 ======================================================#
# current_time = now.strftime("0Y-0M-0D-0H-0Minu")
print("current_time==",current_time)
os.chdir(root)


os.chdir(root)
stage_dir = tar_name
tar_root_dir=os.path.join(root,"save" ) # 手动输入完整路径,需要获取classifier的文件夹名称，path可根据config命名得知
tar_base_dir=os.path.join(stage_dir )

# 这里要修改，把文件夹的名字带上
# base_dir 必须相对于 root_dir 给出
shutil.make_archive(base_name=tar_name, format="gztar", root_dir=tar_root_dir, base_dir=tar_base_dir ) #root/.tar.gz
tar_gz_name =tar_name + ".tar.gz"

# 创建目录
current_stage_dir_drive=os.path.join(drive_root,baseline,method,stage,current_time)
os.makedirs(current_stage_dir_drive,exist_ok=True)


# 拷贝到drive
shutil.copy("%s"%(tar_gz_name),current_stage_dir_drive)
# shutil.copy(os.path.join(root,'configs','train_%s_%s.yaml'%(stage,dataset)),current_stage_dir_drive)
# shutil.copy(os.path.join(root,'train_classifier_cutmix.py'),current_stage_dir_drive)
print("%s 保存位置："%tar_gz_name, current_stage_dir_drive)
# 注意是小写的meta-baseline

# classifier阶段pth在云盘中的位置： /content/drive/MyDrive/meta-baseline/fcanet-inMeta-inClass-cutmix/classifier/2021Y-12M-16D-05H-57Minu/classifier_mini_5_way_5_shot_mini.tar.gz

"""

#=================训练meta=========================#
stage="meta"
dataset="mini"
tag="5_way_5_shot_mini"
filename= "train_%s.py"%(stage)
yamlfilename = "configs/train_%s_%s.yaml"%(stage,dataset) #========================================================================= 3 ======================================================#
# yamlfilename = "configs/train_%s_%s_test.yaml"%(stage,dataset)
tar_name = stage+"_"+dataset+"_"+tag

os.chdir(root)
os.chmod(filename,0o777)
# !chmod +777 train_classifier.py
!python $filename --config $yamlfilename  --name $tar_name

#=================保存meta.tar.gz到云盘=========================#
from datetime import datetime
now = datetime.now() # current date and time
current_time = now.strftime("%YY-%mM-%dD-%HH-%MMinu")  #========================================================================= 4 ======================================================#
# current_time  = "0Y-0M-0D-0H-0Minu"
print("current_time==",current_time)
os.chdir(root)


#=================用epoch-last测试meta=========================#
tag="5_way_5_shot_mini"
filename= "test_few_shot.py"
yamlfilename = "configs/test_few_shot.yaml"

os.chdir(root)
os.chmod(filename,0o777)
!python $filename --shot 5 --test-epochs 10   #========================================================================= 5 ======================================================#
# !python $filename --shot 5 --test-epochs 1

os.chdir(root)
os.chmod(filename,0o777)
!python $filename --shot 1 --test-epochs 10   #========================================================================= 5 ======================================================#
# !python $filename --shot 1 --test-epochs 1

# 压缩
# /content/baseline/meta-baseline-fcanet-inMeta-inClass-bsnet-weight/save/meta_mini-imagenet-5shot_meta-baseline-att-resnet12-wide-att
os.chdir(root)
stage_dir = tar_name
# /content/baseline/meta-baseline-fcanet-inMeta-inClass-cutmix/save/meta_mini_5_way_5_shot_mini
tar_root_dir=os.path.join(root,"save" ) # 一律按此方式命名
tar_base_dir=os.path.join(stage_dir )
# base_dir 必须相对于 root_dir 给出
shutil.make_archive(base_name=tar_name, format="gztar", root_dir=tar_root_dir, base_dir=tar_base_dir ) #root/.tar.gz
tar_gz_name =tar_name + ".tar.gz"




# 创建目录
current_stage_dir_drive=os.path.join(drive_root,baseline,method,stage,current_time)
os.makedirs(current_stage_dir_drive,exist_ok=True)

# 拷贝到drive
shutil.copy("%s"%(tar_gz_name),current_stage_dir_drive)
#shutil.copy(os.path.join(root,'configs','train_%s_%s.yaml'%(stage,dataset)),current_stage_dir_drive)
# shutil.copy(os.path.join(root,'train_meta.py'),current_stage_dir_drive)
print("%s 保存位置："%tar_gz_name, current_stage_dir_drive)
# 注意是小写的meta-baseline

# 最后结束运行
# !kill -9 -1 # 该代码不能结束会话，只能重启