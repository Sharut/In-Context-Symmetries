import hydra
import os
import uuid
import string
import random
import datetime
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("uuid", lambda: ''.join(random.choice(string.ascii_letters)for i in range(10))+ '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")), use_cache=False)

from hydra.core.hydra_config import HydraConfig
import logging
import dataset as ds
import numpy as np
import wandb
import torch
import math
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50
import utils
import sys
import omegaconf
from datetime import date
from models import BranchNetwork, OnlinePredictor
from tqdm import tqdm
import warnings
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

def knn_loop(encoder, train_loader, test_loader, knn_meter_feat, knn_meter_pred, args):
    accuracy = utils.knn_monitor(net=encoder.cuda(),
                           memory_data_loader=train_loader,
                           test_data_loader=test_loader,
                           device='cuda',
                           k=200,
                           knn_meter_feat = knn_meter_feat,
                           knn_meter_pred = knn_meter_pred,
                           hide_progress=True,
                           args = args)

def compute_similarities(z):
        sizes = z.size()  
        z = z.view(sizes[0]//2, 2, sizes[1])
        z1= z[:, 0, :].detach()
        z2= z[:, 1, :].detach()

        all_pos = torch.nn.CosineSimilarity()(z1, z2).abs()
        pos = all_pos.mean()

        z1_and_z2 = torch.cat([z1, z2], dim=0)
        z1_and_z2 = F.normalize(z1_and_z2, dim=-1)

        all_neg = (z1_and_z2 @ z1_and_z2.T)
        neg = all_neg.mean()
        return pos, neg, all_pos

def nt_xent(x, x_proj=None, t=0.5):
    if x_proj is None:
        x_proj = x
    x = F.normalize(x, dim=1); x_proj = F.normalize(x_proj, dim=1)
    x_scores =  (x @ x_proj.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k

    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))
   

def adjust_learning_rate(epochs, warmup_epochs, base_lr, optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


@hydra.main(config_path='./', config_name='config.yml')
def train(args: DictConfig) -> None:
    OmegaConf.set_struct(args, False)                               # Set the configuration to be mutable
    torch.set_num_threads(12)
    
    assert torch.cuda.is_available()
    cudnn.benchmark = True
    cur_dir = HydraConfig.get().runtime.output_dir
    runid = cur_dir.split('/')[-1]


    if not args.debug_mode:
        # Initialize wandb
        wandb.init(entity=args.wandb.entity, project=args.wandb.project, name=args.name, id=runid)
        args_dict = omegaconf.OmegaConf.to_container(args, resolve=True)
        curr_date = date.today().strftime("%B %d, %Y")
        wandb.config.update({"curr_date": curr_date}, allow_val_change=True) 
        wandb.config.update(args_dict, allow_val_change=True) 

    else:
        logger.info('You are in DEBUG mode, so no Wandb logging')
    logger.info(f'Saving run logs in {cur_dir}')

    # Data loading and transformations
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
    args.augmentation_dim = 6 # total dimension of the augmentation vector (four for rotation and two for color)
    assert sum(args.env_ratio) == 1
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True,
                              )


    memory_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
    
    # Model definition
    assert args.backbone in ['resnet18', 'resnet34', 'resnet50']
    base_encoder = eval(args.backbone)
    model = BranchNetwork(base_encoder, 
                projection_dim=args.projection_dim, 
                args=args,
                ).cuda()
    
    # Online Linear probe network 
    online_linear_probe = OnlinePredictor(model.feature_dim, args.projection_dim, n_classes, args).cuda()
    online_optimizer = torch.optim.Adam(
                            online_linear_probe.parameters(),
                            lr = 1e-3,
                            weight_decay=0.000001
                            )

    if args.dataparallel:
        logger.info('Using DataParallel...')
        model = utils.data_parallel(model)

    logger.info('Terminal cmd: {}'.format(" ".join(sys.argv)))
    logger.info('Base model: {}'.format(args.backbone))

    optimizer = torch.optim.Adam(
            model.parameters(),
            args.learning_rate, 
            weight_decay=args.weight_decay,
            )


    # Model training
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        online_linear_probe.train()
        pos_meter = AverageMeter("Pos similarity")              # Positive similarity
        neg_meter = AverageMeter("Neg similarity")              # Negative similarity
        loss_meter = AverageMeter("SimCLR_loss")                # Contrastive Loss 
        loss_ce_feat_meter = AverageMeter("CE_loss_feat")       # Cross entropy loss for feature
        loss_ce_pred_meter = AverageMeter("CE_loss_pred")       # Cross entropy loss for predictor
        acc_feat_meter = AverageMeter("Train ACC feat")         # Online linear probe accuracy for feature
        acc_pred_meter = AverageMeter("Train ACC pred")         # Online linear probe accuracy for predictor
        rotation_loss_meter = AverageMeter("Rotation loss")     # Rotation prediction loss
        rot_r2_meter = AverageMeter("Train rot R2")             # R2 score for rotation prediction
        color_loss_meter = AverageMeter("Color loss")           # Color prediction loss
        color_r2_meter = AverageMeter("Train color R2")         # R2 score for color prediction

        if not args.is_invariant:
            test_at = [0, 1, args.block_size-2, args.block_size-1]  
            equiv_acc_meters = {f'r2_c={c}' : AverageMeter(f'Train Equiv R2 ctxt={c}') for c in test_at}
            equiv_acc_meters_color = {f'r2_c={c}' : AverageMeter(f'Train Equiv color R2 ctxt={c}') for c in test_at}
            inv_acc_meters = {f'r2_c={c}' : AverageMeter(f'Train Invariant R2 ctxt={c}') for c in test_at}
            inv_acc_meters_color = {f'r2_c={c}' : AverageMeter(f'Train Invariant color R2 ctxt={c}') for c in test_at}

        train_bar = tqdm(train_loader)
        args.epoch = epoch

        for it, (x, xv, r, i_latent, y) in enumerate(train_bar, start=(epoch - 1) * len(train_loader)):
            optimizer.zero_grad()
            online_optimizer.zero_grad()
            x = x.cuda(non_blocking=True)                       # First view of the images
            xv = xv.cuda(non_blocking=True)                     # Second view of the images
            r = r.cuda(non_blocking=True)                       # Transformation vector of images
            y = y.cuda(non_blocking=True)                       # Label of the images
            ir = i_latent.cuda(non_blocking=True)               # Latent vector of each view of image representing its pose etc. 

            r_view1 = ir[:, 0, :4]                              # Rotation/pose parameter vector of first view
            r_view2 = ir[:, 1, :4]                              # Rotation/pose parameter vector of second view
            c_view1 = ir[:, 0, 4:]                              # Color parameter vector of first view
            c_view2 = ir[:, 1, 4:]                              # Color parameter vector of second view

            lr = adjust_learning_rate(epochs=args.epochs,
                                    warmup_epochs=args.warmup_epochs,
                                    base_lr=args.learning_rate * args.batch_size / 256,
                                    optimizer=optimizer,
                                    loader=train_loader,
                                    step=it)
            
            this_env = np.random.choice(['rot', 'color', 'rotcolor'], p=args.env_ratio)
            if this_env == 'rot':
                r[:, 4:] = 0
            elif this_env == 'color':
                r[:, :4] = 0
            else:
                pass

            x = torch.stack([x, xv], dim=1)
            sizes = x.size()                                         
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4])   # reshape to (bs*n_views, n_channels, H, W) converts to sequence of (x, x+)
            y = torch.stack([y,y], dim=1).view(-1)
            r_y = torch.stack([r, r], dim=1).view(-1, r.size(-1))   
            r_view2_inv = torch.stack([r_view1, r_view2], dim=1).view(-1, r_view2.size(-1))
            c_view2_inv = torch.stack([c_view1, c_view2], dim=1).view(-1, c_view2.size(-1))
            r_view2 = torch.stack([r_view2, r_view2], dim=1).view(-1, r_view2.size(-1))
            c_view2 = torch.stack([c_view2, c_view2], dim=1).view(-1, c_view2.size(-1))

            with torch.cuda.amp.autocast(enabled=True):
                feature, rep, pred, ema_proj = model(x, r, random_mask=args.random_mask, mask_prob=args.mask_prob)
                loss = nt_xent(pred, t=args.temperature)
                pos, neg, _ = compute_similarities(pred)

                # Online linear probe loss and accuracy for feature representation
                class_logits_feat, class_logits_pred = online_linear_probe(feature, pred)
                loss_ce_feat = F.cross_entropy(class_logits_feat, y)
                acc_feat = (class_logits_feat.argmax(dim=1)==y).float().mean()  
                
                # Online linear probe loss and accuracy for predictor representation
                loss_ce_pred = F.cross_entropy(class_logits_pred, y)
                acc_pred = (class_logits_pred.argmax(dim=1)==y).float().mean()  
                
                # Total loss                
                loss_all = loss + loss_ce_feat + loss_ce_pred

                # Rotation prediction loss optimized jointly with SimSiam and CE Loss
                pred = F.normalize(pred, dim=-1)
                equiv_logits = model.aug_predictor(pred) 
                weight_equiv_loss = args.weight_equiv_loss 

                # Online measure of equivariance with respect to rotation
                if this_env == 'rot' or this_env == 'rotcolor':
                    rotation_loss = F.mse_loss(equiv_logits[:, :4], r_view2)
                    r2 = utils.r2_score(equiv_logits[:, :4], r_view2)
                    rotation_loss_meter.update(rotation_loss.item(), x.size(0))
                    rot_r2_meter.update(r2.item(), x.size(0))            
                    loss_all += weight_equiv_loss * rotation_loss
                
                if this_env == 'color' or this_env == 'rotcolor':
                    color_loss = F.mse_loss(equiv_logits[:, 4:], c_view2)
                    r2 = utils.r2_score(equiv_logits[:, 4:], c_view2)
                    color_loss_meter.update(color_loss.item(), x.size(0))
                    color_r2_meter.update(r2.item(), x.size(0))
                    loss_all += weight_equiv_loss * color_loss

            # Measure of equivariance with context 
            if not args.is_invariant :
                # only calculate equiv results for those been trained on
                if this_env == 'rot' or this_env == 'rotcolor':
                    info_dict = utils.evaluate_with_context(equiv_logits[:, :4], r_view2, args, test_at)
                    for k, v in info_dict.items():
                        equiv_acc_meters[k].update(v, x.size(0) // args.block_size)
                    equiv_results = {k: round(v.avg, 3) for k, v in equiv_acc_meters.items()}
                if this_env == 'color' or this_env == 'rotcolor':
                    info_dict = utils.evaluate_with_context(equiv_logits[:, 4:], c_view2, args, test_at)
                    for k, v in info_dict.items():
                        equiv_acc_meters_color[k].update(v, x.size(0) // args.block_size)
                    equiv_results_color = {k: round(v.avg, 3) for k, v in equiv_acc_meters_color.items()}
                if this_env == 'rot':
                    info_dict = utils.evaluate_with_context(equiv_logits[:, 4:], c_view2_inv, args, test_at)
                    for k, v in info_dict.items():
                        inv_acc_meters[k].update(v, x.size(0) // args.block_size)
                    inv_results_color = {k: round(v.avg, 3) for k, v in inv_acc_meters.items()}
                if this_env == 'color':
                    info_dict = utils.evaluate_with_context(equiv_logits[:, :4], r_view2_inv, args, test_at)
                    for k, v in info_dict.items():
                        inv_acc_meters_color[k].update(v, x.size(0) // args.block_size)
                    inv_results = {k: round(v.avg, 3) for k, v in inv_acc_meters_color.items()}
            else:
                equiv_results = 'NA'
                equiv_results_color = 'NA'
                inv_results = 'NA'
                inv_results_color = 'NA'
                
            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.step(online_optimizer)
            scaler.update()

            # update all average meters and logs
            pos_meter.update(pos.item(), x.size(0))
            neg_meter.update(neg.item(), x.size(0))
            loss_meter.update(loss.item(), x.size(0))
            loss_ce_feat_meter.update(loss_ce_feat.item(), x.size(0))
            loss_ce_pred_meter.update(loss_ce_pred.item(), x.size(0))
            acc_feat_meter.update(acc_feat.item(), x.size(0))
            acc_pred_meter.update(acc_pred.item(), x.size(0))

            desc = f"Train epoch {epoch}, {args.method} loss: {loss_meter.avg:.4f} Rot loss: {rotation_loss_meter.avg:.3f} R2: {rot_r2_meter.avg:.3f} Color loss: {color_loss_meter.avg:.3f} Color R2: {color_r2_meter.avg:.3f} Train ACC feat: {acc_feat_meter.avg:.4f} Train Acc pred: {acc_pred_meter.avg:.4f}"
            train_bar.set_description(desc)
            if not args.debug_mode:
                log_dict = {'Train loss': loss_meter.avg, 
                        'Loss CE feat': loss_ce_feat_meter.avg,
                        'Train acc feat': acc_feat_meter.avg,
                        'Loss CE pred': loss_ce_pred_meter.avg,
                        'Train acc pred': acc_pred_meter.avg,
                        'Pos similarity': pos_meter.avg,
                        'Neg similarity': neg_meter.avg,
                        'Learning rate': optimizer.param_groups[0]['lr'],
                        'Rotation loss': rotation_loss_meter.avg,
                        'Rotation R2': rot_r2_meter.avg,
                        'Color loss': color_loss_meter.avg,
                        'Color R2': color_r2_meter.avg,
                        }
                
                wandb.log(log_dict)
                
        # save checkpoint only at log_interval epochs
        if epoch >= args.log_interval and epoch % args.log_interval == 0:
            # Online linear probe accuracy
            for test_mode in ['rot', 'color']:
                eval_ctxt = [0, args.block_size-2]
                if args.cache_type == 'train':
                    test_logs = utils.evaluation_with_cache(model, online_linear_probe, train_dataset.cache, test_loader, test_mode, eval_ctxt, args)
                else:  
                    test_logs = utils.evaluation_with_cache(model, online_linear_probe, test_dataset.cache, test_loader, test_mode, eval_ctxt, args)
                
                str_test_logs = ", ".join(f"{key}: {value}" for key, value in test_logs.items())

                # kNN accuracy
                knn_top1_meter_feat = AverageMeter('kNN top-1 acc feat')
                knn_top1_meter_pred = AverageMeter('kNN top-1 acc pred')
                knn_loop(model, memory_loader, test_loader, knn_top1_meter_feat, knn_top1_meter_pred, args)

                if test_mode == 'rot':
                    logger.info(f"==> Train epoch {epoch}, Train {args.method} loss: {loss_meter.avg:.4f} Train CE loss feat: {loss_ce_feat_meter.avg:.4f} Train ACC feat: {acc_feat_meter.avg:.4f} Train CE loss pred: {loss_ce_pred_meter.avg:.4f} Train ACC pred: {acc_pred_meter.avg:.4f} Train Rot loss: {rotation_loss_meter.avg:.4f} Train Rot R2: {rot_r2_meter.avg:.4f} Train Color loss: {color_loss_meter.avg:.4f} Train Color R2: {color_r2_meter.avg:.4f} Train Pos Sim: {pos_meter.avg:.4f} Train Neg Sim: {neg_meter.avg:.4f}")
                    logger.info(f"==> Train Equiv rot: {equiv_results} Train Equiv color: {equiv_results_color} Train Inv rot: {inv_results} Train Inv color: {inv_results_color}")
                
                if not args.debug_mode:
                    wandb.log({
                        'Test kNN Acc feat': knn_top1_meter_feat.avg,
                        'Test kNN Acc pred': knn_top1_meter_pred.avg,
                        'Test acc feat': test_logs['Test acc feat']['z+']['c=0'],
                        'Test acc pred': test_logs['Test acc pred']['z+']['c=0'],
                    })

                logger.info(f"==> Test mode={test_mode} Test kNN Acc feat: {knn_top1_meter_feat.avg:.4f} Test kNN Acc pred: {knn_top1_meter_pred.avg:.4f} {str_test_logs}")
          
            if epoch % args.save_interval == 0:
                logger.info(f"==> Save checkpoint at epoch {epoch}...")
                torch.save(model.state_dict(), 'simclr_{}_epoch{}.pt'.format(args.backbone, epoch))
                
            
    wandb.finish()

if __name__ == '__main__':
    train()
