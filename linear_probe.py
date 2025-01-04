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


class OfflineProbe(nn.Module):
    def __init__(self, feature_dim, predictor_dim, n_classes, net='linear', modes=['none'], ctxt_list=[0,126]):
        super(OfflineProbe, self).__init__()
        self.feature_dim = feature_dim
        self.predictor_dim = predictor_dim
        self.n_classes = n_classes
        self.modes = modes
        self.networks = {}
        for mode in modes:
            if net == 'linear':
                feature_clf = nn.Linear(self.feature_dim, n_classes).cuda()
                predictor_zr_clf = nn.ModuleDict({
                    str(ctxt_len): nn.Linear(self.predictor_dim, n_classes).cuda() for ctxt_len in ctxt_list
                })
                predictor_zplus_clf = nn.ModuleDict({
                    str(ctxt_len): nn.Linear(self.predictor_dim, n_classes).cuda() for ctxt_len in ctxt_list
                })
            elif net == 'mlp':
                feature_clf = nn.Sequential(
                                        nn.Linear(self.feature_dim, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024,1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, n_classes),
                                    ).cuda()
                predictor_zr_clf = nn.ModuleDict({
                    str(ctxt_len): nn.Sequential(
                                        nn.Linear(self.predictor_dim, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024,1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, n_classes),
                                    ).cuda() for ctxt_len in ctxt_list
                })
                predictor_zplus_clf = nn.ModuleDict({
                    str(ctxt_len): nn.Sequential(
                                        nn.Linear(self.predictor_dim, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024,1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, n_classes),
                                    ).cuda() for ctxt_len in ctxt_list
                })
            self.networks[mode] = nn.ModuleDict({
                'feature_clf': feature_clf,
                'predictor_zr_clf': predictor_zr_clf,
                'predictor_zplus_clf': predictor_zplus_clf,
            })
        self.networks = nn.ModuleDict(self.networks)


    def forward(self, x, zplus, zr=None, mode='none'):
        feature_logits = self.networks[mode]['feature_clf'](x)
        predictor_zplus_logits = self.networks[mode]['predictor_zplus_clf'](zplus)
        if zr is not None:
            predictor_zr_logits = self.networks[mode]['predictor_zr_clf'](zr)
            return feature_logits, predictor_zplus_logits, predictor_zr_logits
        return feature_logits, predictor_zplus_logits
    
    def forward_feat(self, x, mode='none'):
        feature_logits = self.networks[mode]['feature_clf'](x)
        return feature_logits
    
    def forward_zplus(self, zplus, mode='none', ctxt=0):
        predictor_zplus_logits = self.networks[mode]['predictor_zplus_clf'][str(ctxt)](zplus)
        return predictor_zplus_logits
    
    def forward_zr(self, zr, mode='none', ctxt=0):
        predictor_zr_logits = self.networks[mode]['predictor_zr_clf'][str(ctxt)](zr)
        return predictor_zr_logits
    

@hydra.main(config_path='./', config_name='eval_config.yml')
def train(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False)           ## Set the configuration to be mutable
    torch.set_num_threads(12)
    
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    # set seed
    set_seed(args.seed, use_cuda=True)

    # set up data augmentation
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
    
    eval_dataset = ds.EvalDatasetAll(os.path.join(args.data_dir, '3DIEBench'),
                                                os.path.join(args.data_dir, 'meta', 'val_images.npy'), 
                                                os.path.join(args.data_dir, 'meta', 'val_labels.npy'), 
                                                transform=test_transform,
                                                args=args,)

    n_classes = 55
    args.augmentation_dim = 6
    test_modes = ['rot', 'color', 'none']

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True,
                              )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
    
    assert args.backbone in ['resnet18', 'resnet34', 'resnet50']
    base_encoder = eval(args.backbone)
    model = BranchNetwork(base_encoder, 
                projection_dim=args.projection_dim, 
                args=args,
                ).cuda()

    label_linear_probe = OfflineProbe(model.feature_dim, args.projection_dim, n_classes, modes=test_modes, ctxt_list=args.eval_ctxt).cuda()
    label_optimizer = torch.optim.Adam(label_linear_probe.parameters(), lr=1e-3, betas=(0.9, 0.999))

    color_linear_probe = OfflineProbe(model.feature_dim, args.projection_dim, 2, modes=test_modes, ctxt_list=args.eval_ctxt).cuda()
    color_optimizer = torch.optim.Adam(color_linear_probe.parameters(), lr=1e-3, betas=(0.9, 0.999))

    rot_linear_probe = OfflineProbe(model.feature_dim, args.projection_dim, 4, net='mlp', modes=test_modes, ctxt_list=args.eval_ctxt).cuda()
    rot_optimizer = torch.optim.Adam(rot_linear_probe.parameters(), lr=1e-3, betas=(0.9, 0.999))

    logger.info(f'Saving run logs in {HydraConfig.get().runtime.output_dir}')
    logger.info('Terminal cmd: {}'.format(" ".join(sys.argv)))
    logger.info('Base model: {}'.format(args.backbone))
    logger.info('feature dim: {}, projection dim: {}'.format(model.feature_dim, args.projection_dim))

    model_dir = os.path.join(args.pretrained_model_dir, args.load_model)
    logger.info('Loading model from {}'.format(model_dir))
    model.load_state_dict(torch.load(model_dir))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # kv cache is pre-calculated
    with torch.no_grad():
        cache = train_dataset.cache
        cache_x, cache_r, cache_y = cache
        assert cache_x is not None
        
        past_dict_bymode = {}
        past_dict_bymode_single = {}
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
            past_dict_single = {}
            for ctxt_len in args.eval_ctxt:
                assert ctxt_len%2 ==0, 'So far only even context lengths are supported for evaluation'
                with torch.no_grad():  
                    if ctxt_len == 0:    
                        initial_past = None
                        initial_past_double = None
                        initial_past_single = None
                    else:
                        curr_x = cache_x[:ctxt_len, :, :, :].cuda(non_blocking=True)
                        curr_r = cache_r_input[:ctxt_len//2, :].cuda(non_blocking=True)

                        initial_past = model(curr_x, curr_r, return_cache = True)[-1]
                        initial_past = [[p[:, :, :ctxt_len, :] for p in layer] for layer in initial_past]
                        initial_past_double = utils.repeat_past_key_values(initial_past, args.batch_size*2)
                        initial_past_single = utils.repeat_past_key_values(initial_past, args.batch_size)
                        
                    past_dict[ctxt_len] = initial_past_double
                    past_dict_single[ctxt_len] = initial_past_single
            past_dict_bymode[mode] = past_dict
            past_dict_bymode_single[mode] = past_dict_single

    for epoch in range(1, args.epochs + 1):
        model.eval()
        label_linear_probe.train()
        color_linear_probe.train()
        rot_linear_probe.train()

        train_bar = tqdm(train_loader)
        for it, (x, xv, r, i_latent, y) in enumerate(train_bar, start=(epoch - 1) * len(train_loader)):
            
            x = x.cuda(non_blocking=True)
            xv = xv.cuda(non_blocking=True)
            r = r.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            ir = i_latent.cuda(non_blocking=True)
            


            r_view1 = ir[:, 0, :4]
            r_view2 = ir[:, 1, :4]
            c_view1 = ir[:, 0, 4:]
            c_view2 = ir[:, 1, 4:]

            r_copy = r.clone()

            x = torch.stack([x, xv], dim=1)
            sizes = x.size()            # x is of shape (bs, n_views, n_channels, H, W)
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4])   # reshape to (bs*n_views, n_channels, H, W) converts to sequence of (x, x+)
            y = torch.stack([y,y], dim=1).view(-1)

            r_zplus = torch.stack([r_view1, r_view2], dim=1).view(-1, r_view2.size(-1))
            c_zplus = torch.stack([c_view1, c_view2], dim=1).view(-1, c_view2.size(-1))
            r_zr = torch.stack([r_view2, r_view2], dim=1).view(-1, r_view2.size(-1))
            c_zr = torch.stack([c_view2, c_view2], dim=1).view(-1, c_view2.size(-1))

            if args.is_invariant:
                r_dict = {'none': torch.zeros_like(r_copy)}
                r_tgt_dict = {'none': r_zplus}
                c_tgt_dict = {'none': c_zplus}
            elif args.env_type == 'double':
                rot_r = r_copy.clone()
                color_r = r_copy.clone()
                rot_r[:, 4:] = 0
                color_r[:, :4] = 0
                r_dict = {'rot': rot_r, 'color': color_r, 'none': torch.zeros_like(r_copy)}
                r_tgt_dict = {'rot': r_zr, 'color': r_zplus, 'none': r_zplus}
                c_tgt_dict = {'rot': c_zplus, 'color': c_zr, 'none': c_zplus}

            r_dict = {k: torch.stack([v, torch.zeros_like(v)], dim=1).view(-1, v.size(-1)) for k, v in r_dict.items()}
            output_dict = model.forward_eval_train(x, r_dict, past_dict_bymode)

            for mode, output in output_dict.items():
                feature, zr_dict = output
                feat_logits = label_linear_probe.forward_feat(feature, mode=mode)
                feat_rot = rot_linear_probe.forward_feat(feature, mode=mode)
                feat_color = color_linear_probe.forward_feat(feature, mode=mode)
                
                loss_feat = F.cross_entropy(feat_logits, y) + F.mse_loss(feat_rot, r_zplus) + F.mse_loss(feat_color, c_zplus)
                loss_pred_zr = None
                loss_pred_zplus = None
                y_cp = y.clone()

                for ctxt_len in args.eval_ctxt:
                    zr = zr_dict[ctxt_len][::2]
                    zplus = zr_dict[ctxt_len][1::2]
                    zr_logits = label_linear_probe.forward_zr(zr, mode=mode, ctxt=ctxt_len)
                    zplus_logits = label_linear_probe.forward_zplus(zplus, mode=mode, ctxt=ctxt_len)
                    zr = F.normalize(zr, dim=-1)
                    zplus = F.normalize(zplus, dim=-1)
                    zr_rot = rot_linear_probe.forward_zr(zr, mode=mode, ctxt=ctxt_len)
                    zplus_rot = rot_linear_probe.forward_zplus(zplus, mode=mode, ctxt=ctxt_len)
                    zr_color = color_linear_probe.forward_zr(zr, mode=mode, ctxt=ctxt_len)
                    zplus_color = color_linear_probe.forward_zplus(zplus, mode=mode, ctxt=ctxt_len)
                    if loss_pred_zr is None:
                        loss_pred_zr = F.cross_entropy(zr_logits, y_cp[::2]) + F.mse_loss(zr_rot, r_tgt_dict[mode][::2]) + F.mse_loss(zr_color, c_tgt_dict[mode][::2])
                        loss_pred_zplus = F.cross_entropy(zplus_logits, y_cp[1::2]) + F.mse_loss(zplus_rot, r_zplus[1::2]) + F.mse_loss(zplus_color, c_zplus[1::2])
                    else:
                        loss_pred_zr += F.cross_entropy(zr_logits, y_cp[::2]) + F.mse_loss(zr_rot, r_tgt_dict[mode][::2]) + F.mse_loss(zr_color, c_tgt_dict[mode][::2])
                        loss_pred_zplus += F.cross_entropy(zplus_logits, y_cp[1::2]) + F.mse_loss(zplus_rot, r_zplus[1::2]) + F.mse_loss(zplus_color, c_zplus[1::2])

                label_optimizer.zero_grad()
                color_optimizer.zero_grad()
                rot_optimizer.zero_grad()
                loss = loss_feat + loss_pred_zr + loss_pred_zplus
                loss.backward()
                label_optimizer.step()
                color_optimizer.step()
                rot_optimizer.step()
            train_bar.set_description('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch, args.epochs, loss.item()))
                
        if epoch >= args.log_interval and epoch % args.log_interval == 0:
            logger.info(f'Epoch {epoch}')
            label_linear_probe.eval()
            color_linear_probe.eval()
            rot_linear_probe.eval()
            # eval for z+ with the individual eval_loader
            test_bar = tqdm(eval_loader, desc=f'Eval z+')
            labels, i_latents = [], []
            all_logits_feat_bymode, all_rot_feat_bymode, all_color_feat_bymode = {mode: [] for mode in test_modes}, {mode: [] for mode in test_modes}, {mode: [] for mode in test_modes}
            all_logits_zplus_bymode = {mode: {ctxt_len: [] for ctxt_len in args.eval_ctxt} for mode in test_modes}
            all_rot_zplus_bymode = {mode: {ctxt_len: [] for ctxt_len in args.eval_ctxt} for mode in test_modes}
            all_color_zplus_bymode = {mode: {ctxt_len: [] for ctxt_len in args.eval_ctxt} for mode in test_modes}
            with torch.no_grad():
                for x, i_latent, y in test_bar:
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)
                    i_latent = i_latent.cuda(non_blocking=True)
                    r = torch.zeros((x.size(0), args.augmentation_dim)).cuda(non_blocking=True)
                    labels.append(y.detach().cpu())
                    i_latents.append(i_latent.detach().cpu())

                    for mode in test_modes:
                        feature, pred_zplus_dict = model.forward_eval_rplus(x, r, past_dict_bymode_single[mode])
                        logits_hat_feat = label_linear_probe.forward_feat(feature, mode=mode)
                        rot_hat_feat = rot_linear_probe.forward_feat(feature, mode=mode)
                        color_hat_feat = color_linear_probe.forward_feat(feature, mode=mode)
                        all_logits_feat_bymode[mode].append(logits_hat_feat.detach().cpu())
                        all_rot_feat_bymode[mode].append(rot_hat_feat.detach().cpu())
                        all_color_feat_bymode[mode].append(color_hat_feat.detach().cpu())
                        for ctxt_len in args.eval_ctxt:
                            embd_zplus = pred_zplus_dict[ctxt_len]
                            logits_hat_zplus = label_linear_probe.forward_zplus(embd_zplus, mode=mode, ctxt=ctxt_len)
                            embd_zplus = F.normalize(embd_zplus, dim=-1)
                            rot_hat_zplus = rot_linear_probe.forward_zplus(embd_zplus, mode=mode, ctxt=ctxt_len)
                            color_hat_zplus = color_linear_probe.forward_zplus(embd_zplus, mode=mode, ctxt=ctxt_len)
                            all_logits_zplus_bymode[mode][ctxt_len].append(logits_hat_zplus.detach().cpu())
                            all_rot_zplus_bymode[mode][ctxt_len].append(rot_hat_zplus.detach().cpu())
                            all_color_zplus_bymode[mode][ctxt_len].append(color_hat_zplus.detach().cpu())

            test_logs = {mode: {'Test acc feat': 0, 'Test acc z+': {}, 'Test Rot R2 feat': 0, 'Test Rot R2 z+': {}, 'Test Color R2 feat': 0, 'Test Color R2 z+': {}} for mode in test_modes}
            labels = torch.cat(labels, dim=0)
            i_latents = torch.cat(i_latents, dim=0)
            all_logits_feat_bymode = {mode: torch.cat(all_logits_feat_bymode[mode], dim=0) for mode in test_modes}
            all_rot_feat_bymode = {mode: torch.cat(all_rot_feat_bymode[mode], dim=0) for mode in test_modes}
            all_color_feat_bymode = {mode: torch.cat(all_color_feat_bymode[mode], dim=0) for mode in test_modes}
            logits_acc_feat = {mode: (all_logits_feat_bymode[mode].argmax(dim=1) == labels).float().mean() for mode in test_modes}

            rot_r2_feat = {mode: utils.r2_score(all_rot_feat_bymode[mode], i_latents[:, :4]) for mode in test_modes}
            color_r2_feat = {mode: utils.r2_score(all_color_feat_bymode[mode], i_latents[:, 4:]) for mode in test_modes}
            for mode in test_modes:
                test_logs[mode]['Test acc feat'] = round(logits_acc_feat[mode].item(), 4)
                test_logs[mode]['Test Rot R2 feat'] = round(rot_r2_feat[mode].item(), 4)
                test_logs[mode]['Test Color R2 feat'] = round(color_r2_feat[mode].item(), 4)
                labels_cp = labels.clone()

                for ctxt_len in args.eval_ctxt:
                    all_logits_zplus_bymode[mode][ctxt_len] = torch.cat(all_logits_zplus_bymode[mode][ctxt_len], dim=0)
                    all_rot_zplus_bymode[mode][ctxt_len] = torch.cat(all_rot_zplus_bymode[mode][ctxt_len], dim=0)
                    all_color_zplus_bymode[mode][ctxt_len] = torch.cat(all_color_zplus_bymode[mode][ctxt_len], dim=0)
                    
                    class_acc_zplus = (all_logits_zplus_bymode[mode][ctxt_len].argmax(dim=1) == labels_cp).float().mean()
                    
                    rot_r2_zplus = utils.r2_score(all_rot_zplus_bymode[mode][ctxt_len], i_latents[:, :4])
                    color_r2_zplus = utils.r2_score(all_color_zplus_bymode[mode][ctxt_len], i_latents[:, 4:])
                    test_logs[mode]['Test acc z+'][ctxt_len] = round(class_acc_zplus.item(), 4)
                    test_logs[mode]['Test Rot R2 z+'][ctxt_len] = round(rot_r2_zplus.item(), 4)
                    test_logs[mode]['Test Color R2 z+'][ctxt_len] = round(color_r2_zplus.item(), 4)
            
            if not args.is_invariant and args.env_type in ['double']:
                # eval for zr with test_loader (the paired one)
                test_bar = tqdm(test_loader, desc=f'Eval zr')
                labels = []
                if args.env_type == 'double':
                    test_modes_zr = ['rot', 'color']
                i_latents_bymode_rot = {mode: [] for mode in test_modes_zr}
                i_latents_bymode_color = {mode: [] for mode in test_modes_zr}
                all_logits_zr_bymode = {mode: {ctxt_len: [] for ctxt_len in args.eval_ctxt} for mode in test_modes_zr}


                all_rot_zr_bymode = {mode: {ctxt_len: [] for ctxt_len in args.eval_ctxt} for mode in test_modes_zr}
                all_color_zr_bymode = {mode: {ctxt_len: [] for ctxt_len in args.eval_ctxt} for mode in test_modes_zr}
                with torch.no_grad():
                    for x, xv, r, i_latent, y in test_bar:
                        x, xv = x.cuda(non_blocking=True), xv.cuda(non_blocking=True)
                        y = y.cuda(non_blocking=True)
                        ir = i_latent.cuda(non_blocking=True)
                        r = r.cuda(non_blocking=True)

                        r_view1 = ir[:, 0, :4]
                        r_view2 = ir[:, 1, :4]
                        c_view1 = ir[:, 0, 4:]
                        c_view2 = ir[:, 1, 4:]

                        if args.env_type == 'double':
                            r_moderot = r.clone()
                            r_modecolor = r.clone()
                            r_moderot[:, 4:] = 0
                            r_modecolor[:, :4] = 0
                            r_dict = {'rot': r_moderot, 'color': r_modecolor}
                            i_latents_bymode_rot['rot'].append(r_view2.detach().cpu())
                            i_latents_bymode_color['rot'].append(c_view1.detach().cpu())
                            i_latents_bymode_rot['color'].append(r_view1.detach().cpu())
                            i_latents_bymode_color['color'].append(c_view2.detach().cpu())


                        output_dict = model.forward_eval_train(x, r_dict, past_dict_bymode_single)
                        labels.append(y.detach().cpu())

                        for mode in test_modes_zr:
                            feature, zr_dict = output_dict[mode]
                            for ctxt_len in args.eval_ctxt:
                                zr = zr_dict[ctxt_len]
                                zr_logits = label_linear_probe.forward_zr(zr, mode=mode, ctxt=ctxt_len)
                                all_logits_zr_bymode[mode][ctxt_len].append(zr_logits.detach().cpu())
                                zr = F.normalize(zr, dim=-1)
                                
                                zr_rot = rot_linear_probe.forward_zr(zr, mode=mode, ctxt=ctxt_len)
                                zr_color = color_linear_probe.forward_zr(zr, mode=mode, ctxt=ctxt_len)
                            
                                all_rot_zr_bymode[mode][ctxt_len].append(zr_rot.detach().cpu())
                                all_color_zr_bymode[mode][ctxt_len].append(zr_color.detach().cpu())

                labels = torch.cat(labels, dim=0)
                i_latents_bymode_rot = {mode: torch.cat(i_latents_bymode_rot[mode], dim=0) for mode in test_modes_zr}
                i_latents_bymode_color = {mode: torch.cat(i_latents_bymode_color[mode], dim=0) for mode in test_modes_zr}

                for mode in test_modes_zr:
                    test_logs[mode]['Test acc zr'] = {}
                    test_logs[mode][f'Test Rot R2 zr'] = {}
                    test_logs[mode][f'Test Color R2 zr'] = {}
                    

                for mode in test_modes_zr:
                    labels_cp = labels.clone()

                    for ctxt_len in args.eval_ctxt:
                        all_logits_zr_bymode[mode][ctxt_len] = torch.cat(all_logits_zr_bymode[mode][ctxt_len], dim=0)
                        class_acc_zr = (all_logits_zr_bymode[mode][ctxt_len].argmax(dim=1) == labels_cp).float().mean()

                        all_rot_zr_bymode[mode][ctxt_len] = torch.cat(all_rot_zr_bymode[mode][ctxt_len], dim=0)
                        all_color_zr_bymode[mode][ctxt_len] = torch.cat(all_color_zr_bymode[mode][ctxt_len], dim=0)
                        r2_rot = utils.r2_score(all_rot_zr_bymode[mode][ctxt_len], i_latents_bymode_rot[mode])
                        r2_color = utils.r2_score(all_color_zr_bymode[mode][ctxt_len], i_latents_bymode_color[mode])
                        test_logs[mode]['Test acc zr'][ctxt_len] = round(class_acc_zr.item(), 4)
                        test_logs[mode][f'Test Rot R2 zr'][ctxt_len] = round(r2_rot.item(), 4)
                        test_logs[mode][f'Test Color R2 zr'][ctxt_len] = round(r2_color.item(), 4)
            logger.info(test_logs)

    final_logs = test_logs
    final_logs['epoch'] = args.epochs
    final_logs['seed'] = args.seed
    final_logs['model_dir'] = model_dir
    log_entry = {f'seed_{args.seed}': final_logs}
    log_file_path = os.path.join(HydraConfig.get().runtime.output_dir, 'final_logs_individual_debugedfinal.json')
    with open(log_file_path, 'a') as file:  # Open file in append mode
        json.dump(log_entry, file, indent=4)
        file.write(",\n")  # Add comma and newline for JSON array format


if __name__ == '__main__':
    train()
