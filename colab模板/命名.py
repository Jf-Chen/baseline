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

#======================命名====================#
# 代码
method="fcanet-inMeta-inClass-cutmix-v15"
repository="baseline"
baseline="meta-baseline"
root="/content/%s/%s-%s"%(repository,baseline,method)

# 云盘
drive_root="/content/drive/MyDrive"




import os
import shutil 
import tarfile