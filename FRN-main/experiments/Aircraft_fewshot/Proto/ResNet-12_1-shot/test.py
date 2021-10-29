import sys
import os
import torch
import yaml
sys.path.append('../../../../')
from models.Proto import Proto
from utils import util
from trainers.eval import meta_test


with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path,'Aircraft_fewshot/test_pre')
model_path = './model_ResNet-12.pth'
#model_path = '../../../../trained_model_weights/Aircraft_fewshot/Proto/ResNet-12_1-shot/model.pth'

gpu = 0
torch.cuda.set_device(gpu)

model = Proto(resnet=True)
model.cuda()
model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
model.eval()

with torch.no_grad():
    way = 5
    for shot in [1]:
        mean,interval = meta_test(data_path=test_path,
                                model=model,
                                way=way,
                                shot=shot,
                                pre=True,
                                transform_type=None,
                                trial=10000)
        print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))