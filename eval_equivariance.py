import hydra
import os
import uuid
import string
import json
import random
import datetime
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("uuid", lambda: ''.join(random.choice(string.ascii_letters)for i in range(10))+ '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")), use_cache=False)

from hydra.core.hydra_config import HydraConfig
import logging
import dataset as ds
import numpy as np
from PIL import Image
import wandb
import torch
import math
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms
import utils
import sys
import omegaconf
from datetime import date
from models import BranchNetwork, OnlinePredictor
from utils import set_seed
from tqdm import tqdm
import warnings
import torchvision
import torch.nn as nn
import torchvision.transforms as T
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class OfflinePairProbe(nn.Module):
    def __init__(self, feature_dim, predictor_dim, n_classes, net='linear', modes=['none'], ctxt_list=[0,126]):
        super(OfflinePairProbe, self).__init__()
        self.feature_dim = feature_dim
        self.predictor_dim = predictor_dim
        self.n_classes = n_classes
        self.modes = modes
        self.networks = {}
        for mode in modes:
            if net == 'linear':
                feature_clf = nn.Linear(self.feature_dim*2, n_classes).cuda()
                predictor_zplus_clf = nn.ModuleDict({
                    str(ctxt_len): nn.Linear(self.predictor_dim*2, n_classes).cuda() for ctxt_len in ctxt_list
                })
                # zplus because for testing the r is always zero for paired prediction
            elif net == 'mlp':
                feature_clf = nn.Sequential(
                                        nn.Linear(self.feature_dim*2, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024,1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, n_classes),
                                    ).cuda()
                predictor_zplus_clf = nn.ModuleDict({
                    str(ctxt_len): nn.Sequential(
                                        nn.Linear(self.predictor_dim*2, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024,1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, n_classes),
                                        ).cuda() for ctxt_len in ctxt_list
                                    })
            self.networks[mode] = nn.ModuleDict({
                'feature_clf': feature_clf,
                'predictor_zplus_clf': predictor_zplus_clf,
            })
        self.networks = nn.ModuleDict(self.networks)


    def forward(self, x, zplus, mode='none', ctxt=0):
        feature_logits = self.networks[mode]['feature_clf'](x)
        predictor_zplus_logits = self.networks[mode]['predictor_zplus_clf'][str(ctxt)](zplus)
        return feature_logits, predictor_zplus_logits
    
    def forward_feat(self, x, mode='none'):
        feature_logits = self.networks[mode]['feature_clf'](x)
        return feature_logits
    
    def forward_pred(self, zplus, mode='none', ctxt=0):
        predictor_zplus_logits = self.networks[mode]['predictor_zplus_clf'][str(ctxt)](zplus)
        return predictor_zplus_logits


@hydra.main(config_path='./', config_name='eval_config.yml')
def train(args: DictConfig) -> None:
    print(HydraConfig.get().runtime.output_dir)
    OmegaConf.set_struct(args, False)           ## Set the configuration to be mutable    
    torch.set_num_threads(12)
    assert torch.cuda.is_available()
    cudnn.benchmark = True
    
    # set seed
    set_seed(args.seed, use_cuda=True)

    train_transform = test_transform = transforms.Compose([ 
            transforms.Resize((args.resolution,args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]),
        ])

    train_dataset = ds.Dataset3DIEBenchRotColor(os.path.join(args.data_dir, '3DIEBench'),
                                                os.path.join(args.data_dir, 'meta', 'train_images.npy'), 
                                                os.path.join(args.data_dir, 'meta', 'train_labels.npy'), 
                                                transform=train_transform,
                                                args=args,)
    test_dataset = ds.Dataset3DIEBenchRotColor(os.path.join(args.data_dir, '3DIEBench'),
                                                os.path.join(args.data_dir, 'meta', 'val_images.npy'), 
                                                os.path.join(args.data_dir, 'meta', 'val_labels.npy'), 
                                                transform=test_transform,
                                                args=args,)
    n_classes = 55
    args.augmentation_dim = 6

    if args.is_invariant:
        test_modes = ['none']
    elif args.env_type == 'single':
        test_modes = ['aug', 'none']
    elif args.env_type == 'double':
        test_modes = ['rot', 'color', 'none']

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True,
                              )


    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    # Prepare model
    assert args.backbone in ['resnet18', 'resnet34', 'resnet50']
    base_encoder = eval(args.backbone)
    model = BranchNetwork(base_encoder, 
                projection_dim=args.projection_dim, 
                args=args,
                ).cuda()


    color_linear_probe = OfflinePairProbe(model.feature_dim, args.projection_dim, 2, modes=test_modes, ctxt_list=args.eval_ctxt).cuda()
    color_optimizer = torch.optim.Adam(color_linear_probe.parameters(), lr=1e-3, betas=(0.9, 0.999))

    rot_linear_probe = OfflinePairProbe(model.feature_dim, args.projection_dim, 4, net='mlp', modes=test_modes, ctxt_list=args.eval_ctxt).cuda()
    rot_optimizer = torch.optim.Adam(rot_linear_probe.parameters(), lr=1e-3, betas=(0.9, 0.999))

    
    logger.info(f'Saving run logs in {HydraConfig.get().runtime.output_dir}')
    logger.info('Terminal cmd: {}'.format(" ".join(sys.argv)))
    logger.info('Base model: {}'.format(args.backbone))
    logger.info('feature dim: {}, projection dim: {}'.format(model.feature_dim, args.projection_dim))

    # model.load_state_dict(torch.load(args.pretrained_model_dir))
    model_dir = os.path.join(args.pretrained_model_dir, args.load_model)
    logger.info('Loading model from {}'.format(model_dir))
    model.load_state_dict(torch.load(model_dir))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # kv cache is pre-calculated
    cache = train_dataset.cache
    cache_x, cache_r, cache_y = cache
    assert cache_x is not None


    past_dict_bymode = {}
    # this is the key output dict of cache calculation, with keys being test_modes and values being dicts of ctxt_len to past
    for mode in test_modes:
        cache_r_input = cache_r.clone()
        if mode == 'rot':
            cache_r_input[:, 4:] = 0
        elif mode == 'color':
            cache_r_input[:, :4] = 0
        elif mode == 'none':
            cache_r_input = torch.zeros_like(cache_r_input)

        past_dict = {}
        for ctxt_len in args.eval_ctxt:
            assert ctxt_len%2 ==0, 'So far only even context lengths are supported for evaluation'
            with torch.no_grad():  
                if ctxt_len == 0:    
                    initial_past = None
                else:
                    curr_x = cache_x[:ctxt_len, :, :, :].cuda(non_blocking=True)
                    curr_r = cache_r_input[:ctxt_len//2, :].cuda(non_blocking=True)
                    # print('Testing shapes before caching', cache_x[:ctxt_len, :, :, :].shape, cache_r[:ctxt_len//2, :].shape)
                    initial_past = model(curr_x, curr_r, return_cache = True)[-1]
                    initial_past = [[p[:, :, :ctxt_len, :] for p in layer] for layer in initial_past]
                    initial_past = utils.repeat_past_key_values(initial_past, args.batch_size*2)
                    
                past_dict[ctxt_len] = initial_past
        past_dict_bymode[mode] = past_dict

    # scaler = torch.cuda.amp.GradScaler(enabled=True)
    for epoch in range(1, args.epochs + 1):
        model.eval()
        color_linear_probe.train()
        rot_linear_probe.train()
        
        train_bar = tqdm(train_loader)
        for it, (x, xv, r, i_latent, y) in enumerate(train_bar, start=(epoch - 1) * len(train_loader)):
            
            x = x.cuda(non_blocking=True)
            xv = xv.cuda(non_blocking=True)
            r = r.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            
            rot = r[:, :4]
            color = r[:, 4:]
            r_target = r.clone()
            # r is set to zero for samples not in the cache
            r = torch.zeros_like(r)
            
            # input by pairs
            x = torch.stack([x, xv], dim=1)
            sizes = x.size()            # x is of shape (bs, n_views, n_channels, H, W)
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4])   # reshape to (bs*n_views, n_channels, H, W) converts to sequence of (x, x+)
            r = torch.stack([r, r], dim=1).view(-1, r.size(-1)) # (bs*n_views, 4)

            for mode in test_modes:
                feature, pred_zplus_dict = model.forward_eval_rplus(x, r, past_dict_bymode[mode])
                # pred_zplus_dict is a dict of predictions for all ctxt lengths
                feature = feature.view(feature.size(0)//2, -1)
                rot_hat_feat = rot_linear_probe.forward_feat(feature, mode)
                color_hat_feat = color_linear_probe.forward_feat(feature, mode)
                loss_feat = F.mse_loss(rot_hat_feat, rot) + F.mse_loss(color_hat_feat, color)
                loss_pred = None
                for ctxt_len in args.eval_ctxt:
                    embd_zplus = pred_zplus_dict[ctxt_len]
                    embd_zplus = F.normalize(embd_zplus, dim=-1)
                    embd_zplus = embd_zplus.view(embd_zplus.size(0)//2, -1)
                    rot_hat_pred = rot_linear_probe.forward_pred(embd_zplus, mode, ctxt_len)
                    color_hat_pred = color_linear_probe.forward_pred(embd_zplus, mode, ctxt_len)
                    if loss_pred is None:
                        loss_pred = F.mse_loss(rot_hat_pred, rot) + F.mse_loss(color_hat_pred, color)
                    else:
                        loss_pred += F.mse_loss(rot_hat_pred, rot) + F.mse_loss(color_hat_pred, color)
                color_optimizer.zero_grad()
                rot_optimizer.zero_grad()
                loss = loss_feat + loss_pred
                loss.backward()
                rot_optimizer.step()
                color_optimizer.step()

            train_bar.set_description('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch, args.epochs, loss.item()))
        
        if epoch >= args.log_interval and epoch % args.log_interval == 0:
            logger.info(f'Epoch {epoch}')
            rot_linear_probe.eval()
            color_linear_probe.eval()

            test_bar = tqdm(test_loader, desc='Eval')
            rot_list, color_list = [], []
            all_rot_feat_bymode, all_color_feat_bymode = {mode: [] for mode in test_modes}, {mode: [] for mode in test_modes}
            all_rot_pred_bymode, all_color_pred_bymode = {mode: {ctxt_len: [] for ctxt_len in args.eval_ctxt} for mode in test_modes}, {mode: {ctxt_len: [] for ctxt_len in args.eval_ctxt} for mode in test_modes}
            with torch.no_grad():
                for x, xv, r, i_latent, y in test_bar:
                    
                    x = x.cuda(non_blocking=True)
                    xv = xv.cuda(non_blocking=True)
                    r = r.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)
                    
                    rot = r[:, :4]
                    color = r[:, 4:]
                    r_target = r.clone()
                    r = torch.zeros_like(r)

                    rot_list.append(rot.detach().cpu())
                    color_list.append(color.detach().cpu())

                    x = torch.stack([x, xv], dim=1)
                    sizes = x.size()            # x is of shape (bs, n_views, n_channels, H, W)
                    x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4])   # reshape to (bs*n_views, n_channels, H, W) converts to sequence of (x, x+)
                    r = torch.stack([r, r], dim=1).view(-1, r.size(-1)) # (bs*n_views, 4)
                    
                    for mode in test_modes:
                        feature, pred_zplus_dict = model.forward_eval_rplus(x, r, past_dict_bymode[mode])
                        feature = feature.view(feature.size(0)//2, -1)
                        rot_hat_feat = rot_linear_probe.forward_feat(feature, mode)
                        color_hat_feat = color_linear_probe.forward_feat(feature, mode)
                        all_rot_feat_bymode[mode].append(rot_hat_feat.detach().cpu())
                        all_color_feat_bymode[mode].append(color_hat_feat.detach().cpu())

                        for ctxt_len in args.eval_ctxt:
                            embd_zplus = pred_zplus_dict[ctxt_len]
                            embd_zplus = F.normalize(embd_zplus, dim=-1)
                            embd_zplus = embd_zplus.view(embd_zplus.size(0)//2, -1)
                            rot_hat_pred = rot_linear_probe.forward_pred(embd_zplus, mode, ctxt_len)
                            color_hat_pred = color_linear_probe.forward_pred(embd_zplus, mode, ctxt_len)
                            all_rot_pred_bymode[mode][ctxt_len].append(rot_hat_pred.detach().cpu())
                            all_color_pred_bymode[mode][ctxt_len].append(color_hat_pred.detach().cpu())
                
                test_logs = {mode: {'Test Rot R2 feat': 0, 'Test Rot R2 z+': {}, 'Test Color R2 feat': 0, 'Test Color R2 z+': {}} for mode in test_modes}
                
                all_rot_feat_bymode = {mode: torch.cat(all_rot_feat_bymode[mode], dim=0) for mode in test_modes}
                all_color_feat_bymode = {mode: torch.cat(all_color_feat_bymode[mode], dim=0) for mode in test_modes}
                all_rot = torch.cat(rot_list, dim=0); all_color = torch.cat(color_list, dim=0)
                rot_r2_feat = {mode: utils.r2_score(all_rot_feat_bymode[mode], all_rot) for mode in test_modes}
                color_r2_feat = {mode: utils.r2_score(all_color_feat_bymode[mode], all_color) for mode in test_modes}
                for mode in test_modes:
                    test_logs[mode]['Test Rot R2 feat'] = round(rot_r2_feat[mode].item(), 4)
                    test_logs[mode]['Test Color R2 feat'] = round(color_r2_feat[mode].item(), 4)
                    for ctxt_len in args.eval_ctxt:
                        all_rot_pred_bymode[mode][ctxt_len] = torch.cat(all_rot_pred_bymode[mode][ctxt_len], dim=0)
                        all_color_pred_bymode[mode][ctxt_len] = torch.cat(all_color_pred_bymode[mode][ctxt_len], dim=0)
                        rot_r2_pred = utils.r2_score(all_rot_pred_bymode[mode][ctxt_len], all_rot)
                        color_r2_pred = utils.r2_score(all_color_pred_bymode[mode][ctxt_len], all_color)
                        test_logs[mode]['Test Rot R2 z+'][ctxt_len] = round(rot_r2_pred.item(), 4)
                        test_logs[mode]['Test Color R2 z+'][ctxt_len] = round(color_r2_pred.item(), 4)
                logger.info(test_logs)
    if not args.debug_mode:
        final_logs = test_logs
        final_logs['epoch'] = args.epochs
        final_logs['seed'] = args.seed
        final_logs['model_dir'] = model_dir
        log_entry = {f'seed_{args.seed}': final_logs}
        log_file_path = os.path.join(HydraConfig.get().runtime.output_dir, 'final_logs_debugedfinal.json')
        with open(log_file_path, 'a') as file:  # Open file in append mode
            json.dump(log_entry, file, indent=4)
            file.write(",\n")  # Add comma and newline for JSON array format


if __name__ == '__main__':
    train()
