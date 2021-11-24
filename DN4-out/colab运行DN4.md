!git clone https://github.com/WenbinLee/DN4.git

cd DN4

打开/content/DN4/DN4_Train_5way1shot.py
修改82行，opt,unknown = parser.parse_known_args()
修改270行，correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)

cd /content/DN4/dataset/miniImageNet

下载minist
from google_drive_downloader import GoogleDriveDownloader as gdd
gdd.download_file_from_google_drive(file_id='1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk',
                                    dest_path='./mnist.zip',
                                    unzip=True)
									

解压
!unzip -uq "/content/DN4/mnist.zip" -d "/content/DN4/dataset/miniImageNet"

修改部分参数
!python DN4_Train_5way1shot.py --dataset_dir '/content/DN4/dataset/miniImageNet'  --episodeSize 8 --testepisodeSize 8 --lr  0.04
或者不修改
!python DN4_Train_5way1shot.py --dataset_dir '/content/DN4/dataset/miniImageNet'
（预计运行时间8小时）

还没找出如何打出缩进符号

——————————————————————

需要修改代码，每个epoch执行结束后保存到google drive中