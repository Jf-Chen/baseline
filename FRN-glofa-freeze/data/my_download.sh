#!/bin/bash

. ../utils/parse_yaml.sh
. ../utils/gdownload.sh
. ../utils/conditional.sh

eval $(parse_yaml ../config.yml)
echo 'this is the data_path you are trying to download data into:'
echo $data_path

cd $data_path

# 作者提供了数据包https://drive.google.com/drive/folders/1gHt-Ynku6Yc3mz6aKVTppIfNmzML1sNG





# this section is for downloading the mini-ImageNet
# md5sum for the downloaded mini-ImageNet.tar should be 13fda464dcd4d283e953bfb6633176b8
echo "downloading mini-ImageNet..."
gdownload 1MfEd5MZlgO6lhrigCaKfxxLAsUoaDtMw mini-ImageNet.tar
conditional_tar mini-ImageNet.tar 13fda464dcd4d283e953bfb6633176b8



echo ""