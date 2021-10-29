%cd /content
!echo "下载代码"
!git clone https://github.com/Jf-Chen/FRN-main.git
%cd /content/FRN-main/data
!echo "下载完成"

!echo "下载mini-ImageNet.tar 并完成配置"
%cd /content/FRN-main/data
!chmod 755 /content/FRN-main/data/my_download.sh # 
# 下载数据，下载位置写在config.yml中
!/content/FRN-main/data/my_download.sh
!echo "下载、配置完成"

!echo "下载已经训练好的pre-train model"
%cd /content/FRN-main/trained_model_weights
!chmod 755 /content/FRN-main/trained_model_weights/download_weights.sh
!/content/FRN-main/trained_model_weights/download_weights.sh
!echo "预训练权重下载完成"

!echo "准备环境"
!pip3 install tensorboardX
!echo "tensorboardX已安装"