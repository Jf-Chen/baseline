import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from datasets import dataloaders
from tqdm import tqdm
import pdb

def get_score(acc_list):

    mean = np.mean(acc_list)
    interval = 1.96*np.sqrt(np.var(acc_list)/len(acc_list))

    return mean,interval


def meta_test(data_path,model,way,shot,pre,transform_type,query_shot=16,trial=10000,return_list=False):

    eval_loader = dataloaders.meta_test_dataloader(data_path=data_path,
                                                way=way,
                                                shot=shot,
                                                pre=pre,
                                                transform_type=transform_type,
                                                query_shot=query_shot,
                                                trial=trial)
    
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()

    acc_list = []
    list_dist=torch.empty(0,way)
    list_max_index=[]
    # target始终不变

    for i, (inp,_) in tqdm(enumerate(eval_loader),position=0, leave=True):

        inp = inp.cuda()
        max_index = model.meta_test(inp,way=way,shot=shot,query_shot=query_shot)
        
        #------------------------观察距离------------------------------#
        dist=model.out_neg_l2_dist(inp,way=way,shot=shot,query_shot=query_shot)
        
        pdb.set_trace()
        # inp [105,3,84,84]
        # max_index [80]
        # dist [80,5]
        # target[80]
        
        list_dist=torch.cat((list_dist,dist),0)
        list_max_index.append(max_index)
        
        
        #------------------------end------------------------------#

        acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way
        acc_list.append(acc)

    if return_list:
        return np.array(acc_list)
    else:
        mean,interval = get_score(acc_list) 
        
        # return mean,interval # 原本的返回值
        #------------------------------------------#
        return mean,interval,list_dist,list_max_index,target
        #------------------------------------------#