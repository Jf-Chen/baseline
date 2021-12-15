import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
import pdb

def main(config):
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
                config['train_dataset'], config['n_shot'])
        svname += '_' + config['model'] + '-' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))
    
    
    #----部分参数添加到yaml-----#
    num_workers=8 # num_workers
    if config.get('num_workers'):
        num_workers = config['num_workers']
    pin_memory = True
    if config.get("pin_memory"):
        pin_memory=config['pin_memory']
    #---------end----------#

    #### Dataset ####

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    train_sampler = CategoriesSampler(
            train_dataset.label, config['train_batches'],
            n_train_way, n_train_shot + n_query,
            ep_per_batch=ep_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)

    # tval
    if config.get('tval_dataset'):
        tval_dataset = datasets.make(config['tval_dataset'],
                                     **config['tval_dataset_args'])
        utils.log('tval dataset: {} (x{}), {}'.format(
                tval_dataset[0][0].shape, len(tval_dataset),
                tval_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
        tval_sampler = CategoriesSampler(
                tval_dataset.label, 200,
                n_way, n_shot + n_query,
                ep_per_batch=4)
        tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                 num_workers=num_workers, pin_memory=pin_memory)
    else:
        tval_loader = None

    # val
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),
            val_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
            val_dataset.label, 200,
            n_way, n_shot + n_query,
            ep_per_batch=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=num_workers, pin_memory=pin_memory)

    ########

    #### Model and optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

    ########
    # device = torch.device('cuda:0') 
    # r_cos = nn.Parameter(torch.ones(1).to(device),requires_grad=True)
    # r_dn4 = nn.Parameter(torch.ones(1).to(device),requires_grad=True)
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va','r_dn4','r_cos','loss_cos','loss_dn4']# ,'logits_dn4_0_0']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model) 
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        np.random.seed(epoch)
        for data, _ in tqdm(train_loader, desc='train', leave=False):
            # 在这里插入样本扩增
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_train_way, n_train_shot, n_query,
                    ep_per_batch=ep_per_batch)
            label = fs.make_nk_label(n_train_way, n_query,
                    ep_per_batch=ep_per_batch).cuda()

            
            #==========================================================================#
            logits_cos_unview,logits_dn4_unview,r_cos,r_dn4 =  model(x_shot, x_query)
            logits_cos = logits_cos_unview.view(-1, n_train_way)
            logits_dn4 = logits_dn4_unview.view(-1, n_train_way)
            logits = logits_cos # 仅仅将logits_dn4作为抑制项
            acc = utils.compute_acc(logits, label)
            
            loss_cos = F.cross_entropy(logits_cos, label)
            loss_dn4 = F.cross_entropy(logits_dn4, label)
            
            loss = 1/(2*r_cos*r_cos)*loss_cos + 1/(r_dn4*r_dn4)*loss_dn4+torch.log(r_cos)+torch.log(r_dn4)
            
            """
            logits_dn4,logits_cos = model(x_shot, x_query)# .view(-1, n_train_way)
            # F.cross_entropy(A, label),A应当是[300,5],;label应当是[300]
            
            logits_dn4_view=logits_dn4.view(-1, n_train_way)
            logits_cos_view=logits_cos.view(-1, n_train_way)
            
            # 理所当然地，二者的logits差距很大，但不影响loss的计算
            # 问题是如何计算acc
            # 应该保持train_meta不变，保持model的输出；先不这样做
            
            loss_dn4 = F.cross_entropy(logits_dn4_view, label)
            loss_cos = F.cross_entropy(logits_cos_view, label)
            loss = logits_dn4 + logits_cos
            # acc = utils.compute_acc(logits, label)
            acc = utils.compute_acc_loss(loss, label)
            """
            #=============================================================================#


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)
            aves['r_dn4'].add(r_dn4.item())
            aves['r_cos'].add(r_cos.item())
            aves['loss_cos'].add(loss_cos.item())
            aves['loss_dn4'].add(loss_dn4.item())
            # aves['logits_dn4_0_0'].add(


            logits = None; loss = None 
            # logits_dn4 = None; logits_cos = None; loss = None 

        # eval
        model.eval()

        for name, loader, name_l, name_a ,name_loss_cos in [
                ('tval', tval_loader, 'tvl', 'tva','loss_cos'),
                ('val', val_loader, 'vl', 'va','loss_cos')]:

            if (config.get('tval_dataset') is None) and name == 'tval':
                continue

            np.random.seed(0)
            for data, _ in tqdm(loader, desc=name, leave=False):
                x_shot, x_query = fs.split_shot_query(
                        data.cuda(), n_way, n_shot, n_query,
                        ep_per_batch=4)
                label = fs.make_nk_label(n_way, n_query,
                        ep_per_batch=4).cuda()

                with torch.no_grad():
                    
                    #==========================================================================#
                    logits_cos_unview,logits_dn4_unview,r_cos,r_dn4 =  model(x_shot, x_query)
                    logits_cos = logits_cos_unview.view(-1, n_way)
                    logits_dn4 = logits_dn4_unview.view(-1, n_way)
                    logits = logits_cos # 仅仅将logits_dn4作为抑制项
                    acc = utils.compute_acc(logits, label)
                    
                    loss_cos = F.cross_entropy(logits_cos, label)
                    loss_dn4 = F.cross_entropy(logits_dn4, label)
                    
                    loss = 1/(2*r_cos*r_cos)*loss_cos + 1/(r_dn4*r_dn4)*loss_dn4+torch.log(r_cos)+torch.log(r_dn4)
                
                aves[name_l].add(loss.item())
                aves[name_a].add(acc)
                aves[name_loss_cos].add(loss_cos.item())

        _sig = int(_[-1])

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
                'val {:.4f}|{:.4f}, {} {}/{} (@{}),r_dn4 {: .4f}, r_cos {: .4f}'.format(
                epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
                aves['vl'], aves['va'], t_epoch, t_used, t_estimate, _sig
                , aves['r_dn4'],aves['r_cos']))
        # 应该把logits的数值也打印出来

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
            'loss_cos':aves['loss_cos'],
            'loss_dn4':aves['loss_dn4'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)
        
        writer.add_scalars('weight', {
            'r_dn4': aves['r_dn4'],
            'r_cos': aves['r_cos'],
        }, epoch)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

