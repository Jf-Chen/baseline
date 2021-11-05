# 需要装载drive
# 指定路径
pathName="baseline/FRN-main-pooling"
data_path="/content/%s/data_path" % pathName
modelPath_inDrive="/content/drive/MyDrive/frn/model_ResNet-12.pth"

%cd /content
!echo "下载代码"
!git clone https://github.com/Jf-Chen/baseline.git
%cd /content/$pathName
!echo "下载完成"

#------------从云盘装载数据集------------#
!echo "从google drive装载 数据集"
%cd $data_path
!cp /content/drive/MyDrive/FRN_fewshot_datasets/mini-ImageNet.tar $data_path
!tar -xf $data_path/mini-ImageNet.tar
!echo "data_path/mini-ImageNet 装载、解压完成"
#---------------end------------------------



!echo "装载已经训练好的fine-tune model"
!echo "模型是30epoch FRN-main微调得到"
%cd /content/$pathName
!cp $modelPath_inDrive /content/$pathName# 来自My Drive/frn/model_ResNet-12.pth # https://drive.google.com/file/d/1-0q-XaI4CGqoWH3UtWbsZ3WRa4AcRGHV/view?usp=sharing
!echo "finetune模型下载完成"

!echo "准备环境"
!pip3 install tensorboardX
!echo "tensorboardX已安装"