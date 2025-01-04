import hydra
import os
import uuid
import string
import json
import random
import datetime
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("uuid", lambda: ''.join(random.choice(string.ascii_letters)for i in range(10))+ '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")), use_cache=False)
import yaml
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
import h5py
import torchvision.transforms as T
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def save_dict_to_hdf5(d, group):
    for key, item in d.items():
        # Convert key to string if it's not already
        if not isinstance(key, str):
            key = str(key)
        
        if isinstance(item, dict):
            subgroup = group.create_group(key)
            save_dict_to_hdf5(item, subgroup)
        elif isinstance(item, np.ndarray):
            group.create_dataset(key, data=item)
        elif torch.is_tensor(item):
            group.create_dataset(key, data=item.numpy())
        else:
            raise ValueError(f"Unsupported data type: {type(item)}")


def load_dict_from_hdf5(group):
    d = {}
    for key, item in group.items():
        # Convert key back to integer if possible
        try:
            int_key = int(key)
        except ValueError:
            int_key = key
        
        if isinstance(item, h5py.Group):
            d[int_key] = load_dict_from_hdf5(item)
        else:
            d[int_key] = np.array(item)
    return d


@hydra.main(config_path='./', config_name='eval_config.yml')
def train(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False)           ## Set the configuration to be mutable
    torch.set_num_threads(12)
    
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    # set seed
    set_seed(args.seed, use_cuda=True)

    test_transform = transforms.Compose([ 
            transforms.Resize((args.resolution,args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]),
        ])

    train_dataset = ds.Dataset3DIEBenchRotColor(os.path.join(args.data_dir, '3DIEBench'),
                                                os.path.join(args.data_dir, 'meta', 'train_images.npy'), 
                                                os.path.join(args.data_dir, 'meta', 'train_labels.npy'), 
                                                transform=test_transform,
                                                args=args,)
    ds_train = ds.EvalDatasetAll(os.path.join(args.data_dir, '3DIEBench'),
                                os.path.join(args.data_dir, 'meta', 'train_images.npy'), 
                                os.path.join(args.data_dir, 'meta', 'train_labels.npy'), 
                                transform=test_transform,
                                args=args,)
    ds_val = ds.EvalDatasetAll(os.path.join(args.data_dir, '3DIEBench'),
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
        test_modes = ['rot', 'color']

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=10,drop_last=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=10,drop_last=True)

    assert args.backbone in ['resnet18', 'resnet34', 'resnet50']
    base_encoder = eval(args.backbone)
    model = BranchNetwork(base_encoder, 
                projection_dim=args.projection_dim, 
                args=args,
                ).cuda()

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
        # 
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

    
    model.eval()
    for loader, name in [(loader_train,"train"), (loader_val,"val")] :
        if os.path.exists(f'{args.pretrained_model_dir}/predictor_embds_{name}.h5'):
            print(f"Feature extraction for the {name} set already done, skipping")
            continue

        feature_embd = {mode: [] for mode in test_modes}
        predictor_embd = {mode: {ctxt_len: [] for ctxt_len in args.eval_ctxt} for mode in test_modes}
        latents = {mode: [] for mode in test_modes}
        bar = tqdm(loader, desc=f'loader={name}')
        print(f"Extracting features for the {name} set ....")
        with torch.no_grad():
            for ind, (x, i_latent, y) in enumerate(bar):
                x = x.cuda(non_blocking=True)
                i_latent = i_latent.cuda(non_blocking=True)
                r = torch.zeros((x.size(0), args.augmentation_dim)).cuda(non_blocking=True)
                for mode in test_modes:
                    latents[mode].append(i_latent.cpu().numpy())
                    feature, pred_zplus_dict = model.forward_eval_rplus(x, r, past_dict_bymode_single[mode])
                    if args.eval_normalize:
                        feature = F.normalize(feature, dim=-1)
                    feature_embd[mode].append(feature.detach().cpu().numpy())
                    for ctxt_len in args.eval_ctxt:
                        embd_zplus = pred_zplus_dict[ctxt_len]
                        if args.eval_normalize:
                            embd_zplus = F.normalize(embd_zplus, dim=-1)
                        predictor_embd[mode][ctxt_len].append(embd_zplus.detach().cpu().numpy())
        all_latents = {mode: np.concatenate(latents[mode]) for mode in test_modes}
        all_feature_embds = {mode: np.concatenate(feature_embd[mode]) for mode in test_modes}
        all_predictor_embds = {mode: {ctxt_len: np.concatenate(predictor_embd[mode][ctxt_len]) for ctxt_len in args.eval_ctxt} for mode in test_modes}
        print(f'Saving embeddings for {name}...')

        with h5py.File(f'{args.pretrained_model_dir}/all_latents_{name}.h5', 'w') as f:
            save_dict_to_hdf5(all_latents, f)
        with h5py.File(f'{args.pretrained_model_dir}/features_embds_{name}.h5', 'w') as f:
            save_dict_to_hdf5(all_feature_embds, f)
        with h5py.File(f'{args.pretrained_model_dir}/predictor_embds_{name}.h5', 'w') as f:
            save_dict_to_hdf5(all_predictor_embds, f)
        print(f'Saved embeddings for {name}...')
    
    model.eval()

    # for source,target in [("train","train"),("val","val")]:
    # for source,target in [("val","val")]:
    for source,target in [("train","train")]:
        print(f"Evaluating {source}-{target}")

        with h5py.File(f'{args.pretrained_model_dir}/predictor_embds_{target}.h5', 'r') as f:
            embeddings_target = load_dict_from_hdf5(f)
        print('Loaded saved embeddings...')
        
        dataset = ds.MRREvalDatasetAll(os.path.join(args.data_dir, '3DIEBench'),
                                os.path.join(args.data_dir, 'meta', f'{source}_images.npy'), 
                                os.path.join(args.data_dir, 'meta', f'{source}_labels.npy'), 
                                transform=test_transform,
                                args=args,)
        
        dataloader = DataLoader(dataset,batch_size=args.batch_size,num_workers=10,shuffle=True,drop_last=True)
        test_bar = tqdm(dataloader, desc=f'Eval zr')
        correct_ranks = {mode: {ctxt_len: [] for ctxt_len in args.eval_ctxt} for mode in test_modes}

        with torch.no_grad():
            for ind, (x, xv, r, start, end, labelidx, local_start, local_end) in enumerate(test_bar):
                x, xv = x.cuda(non_blocking=True), xv.cuda(non_blocking=True)
                r = r.cuda(non_blocking=True)
                
                if args.env_type == 'double':
                    r_moderot = r.clone()
                    r_modecolor = r.clone()
                    r_moderot[:, 4:] = 0
                    r_modecolor[:, :4] = 0
                    r_dict = {'rot': r_moderot, 'color': r_modecolor}
                else:
                    raise NotImplementedError
                
                output_dict = model.forward_eval_train(x, r_dict, past_dict_bymode_single)

                for mode in test_modes:
                    feature, zr_dict = output_dict[mode]
                    for ctxt_len in args.eval_ctxt:
                        zr = zr_dict[ctxt_len]
                        if args.eval_normalize:
                            zr = F.normalize(zr, dim=-1)
                        equi = torch.Tensor(embeddings_target[mode][ctxt_len]).to("cuda:0")
                        for i,out in enumerate(zr):
                            if start[i] + 50 > len(equi) :
                                continue
                            sims = out@equi[start[i]:start[i]+50].T
                            target_idx  = end[i] - start[i]
                            nns = torch.argsort(-sims).cpu()
                            correct_rank = torch.argwhere(nns == target_idx)[0][0]+1
                            correct_ranks[mode][ctxt_len].append(correct_rank)
            correct_ranks_all = {mode: {ctxt_len: torch.stack(correct_ranks[mode][ctxt_len]) for ctxt_len in args.eval_ctxt} for mode in test_modes}
            for mode in test_modes:
                print(f'Mode: {mode}')
                for ctxt_len in args.eval_ctxt:
                    correct_ranks = correct_ranks_all[mode][ctxt_len]
                    MRR = torch.mean(1/correct_ranks)
                    H_at_1 = (correct_ranks <= 1).sum()/correct_ranks.shape[0]
                    H_at_2 = (correct_ranks <= 2).sum()/correct_ranks.shape[0]
                    H_at_5 = (correct_ranks <= 5).sum()/correct_ranks.shape[0]
                    H_at_10 = (correct_ranks <= 10).sum()/correct_ranks.shape[0]
                    print('[Context Length: {}], MRR: {:.4f}, H@1: {:.4f}, H@2: {:.4f}, H@5: {:.4f}, H@10: {:.4f}'.format(ctxt_len, MRR, H_at_1, H_at_2, H_at_5, H_at_10))

if __name__ == '__main__':
    train()