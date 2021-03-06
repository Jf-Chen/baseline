#!/bin/bash
python train.py \
     --opt sgd \
     --lr 1e-3 \
     --gamma 1e-1 \
     --epoch 150 \
     --decay_epoch 70 120 \
     --val_epoch 5 \
     --weight_decay 5e-4 \
     --nesterov \
     --train_transform_type 0 \
     --resnet \
     --train_shot 5 \
     --train_way 20 \
     --test_shot 1 5 \
     --pre \
     --gpu 0
	 
# !python train.py --opt sgd --lr 1e-3 --gamma 1e-1 --epoch 150 --decay_epoch 70 120 --val_epoch 5 --weight_decay 5e-4 --nesterov --train_transform_type 0 --resnet --train_shot 5 --train_way 20 --test_shot 1 5 --pre --gpu 0

