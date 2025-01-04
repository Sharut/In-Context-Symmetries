import wandb
import numpy as np
import torch
import random
import os
from datetime import date
import torch.nn.functional as F
from tqdm import tqdm
import time


def set_seed(seed, use_cuda):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f'=> Seed of the run set to {seed}')

def data_parallel(model):
    if torch.cuda.device_count() >= 1:
        return torch.nn.DataParallel(model)


class  Logger(object):
    def __init__(self, args, tags=None):
        super(Logger, self).__init__()
        print("Project is", args.project)
        self.args=args
        tags=[args.user, tags] if tags is not None else [args.user]
        if args.resume:
            self.run = wandb.init(project=args.project, id = args.run_id, entity=args.user, resume="must", tags=tags)
        elif not args.debug:
            self.run = wandb.init(project=args.project, name = self.args.run_name, entity=args.user, reinit=True, tags=tags)
        config = wandb.config 
        curr_date = date.today()
        curr_date = curr_date.strftime("%B %d, %Y")
        wandb.config.update({"curr_date": curr_date}, allow_val_change=True) 
        wandb.config.update(args, allow_val_change=True) 
           

    def log(self, info):
        if not self.args.debug:
            wandb.log(info)


    def finish(self):
        if not self.args.debug:
            self.run.finish()


# BYOL utility functions
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader, device='cuda', k=200, t=0.1, knn_meter_feat = None, knn_meter_pred = None, hide_progress=False,
                targets=None, args = None):
    if not targets:
        targets = memory_data_loader.dataset.labels
        
    net.eval()
    # classes = len(memory_data_loader.dataset.classes)
    classes = len(set(targets))
    total_top1_feat, total_top1_pred, total_num, feature_bank_feat, feature_bank_pred = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, datav, r_mem, i_latent, target in memory_data_loader:
            data = data.cuda(non_blocking=True)
            datav = datav.cuda(non_blocking=True)
            r_mem = r_mem.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            data = torch.stack([data, datav], dim=1)
            r_mem = torch.zeros_like(r_mem)
            # 
            sizes = data.size()            # data is of shape (bs, n_views, n_channels, H, W)
            data = data.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4])#.cuda(non_blocking=True)   # reshape to (bs*n_views, n_channels, H, W) converts to sequence of (x, x+)
            # target = torch.stack([target,target], dim=1).view(-1).cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=True):
                feature, projection, prediction, ema_projection= net(data, r_mem)
                feature_bank_feat.append(F.normalize(feature, dim=1))
                feature_bank_pred.append(F.normalize(prediction, dim=1))

        # [D, N]
        feature_bank_feat = torch.cat(feature_bank_feat, dim=0).t().contiguous()
        feature_bank_pred = torch.cat(feature_bank_pred, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank_feat.device)
        feature_labels = torch.stack([feature_labels,feature_labels], dim=1).view(-1).cuda(non_blocking=True)
        
        # loop test data to predict the label by weighted knn search
        for data, datav, r_test, i_latent, target in test_data_loader:
            data = data.cuda(non_blocking=True)
            datav = datav.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            r_test = r_test.cuda(non_blocking=True)


            r_test = torch.zeros_like(r_test)

            data = torch.stack([data, datav], dim=1)
            # data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            sizes = data.size()            # data is of shape (bs, n_views, n_channels, H, W)
            data = data.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4])#.cuda(non_blocking=True)   # reshape to (bs*n_views, n_channels, H, W) converts to sequence of (x, x+)
            target = torch.stack([target,target], dim=1).view(-1)#.cuda(non_blocking=True)
            # 
            with torch.cuda.amp.autocast(enabled=True):
                feature, projection, prediction, ema_projection  = net(data, r_test)
                z_feat = F.normalize(feature, dim=1)
                z_pred = F.normalize(prediction, dim=1)
            pred_labels_feat = knn_predict(z_feat, feature_bank_feat, feature_labels, classes, k, t)
            pred_labels_pred = knn_predict(z_pred, feature_bank_pred, feature_labels, classes, k, t)
            total_num += data.size(0)
            total_top1_feat += (pred_labels_feat[:, 0] == target).float().sum().item()
            total_top1_pred += (pred_labels_pred[:, 0] == target).float().sum().item()
            acc_feat = (pred_labels_feat[:, 0] == target).float().mean()
            acc_pred = (pred_labels_pred[:, 0] == target).float().mean()
            knn_meter_feat.update(acc_feat.item(), data.size(0))
            knn_meter_pred.update(acc_pred.item(), data.size(0))

    return total_top1_feat / total_num * 100, total_top1_pred / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

# utility function for creating key-value cache
# Refer to https://github.com/facebookresearch/ICRM/blob/main/algorithms.py
def repeat_past_key_values(past_key_values, repeats):                     # process key value cache for computing fast inference
        repeated_past_key_values = []
        for layer_past in past_key_values:
            repeated_layer_past = []
            for tensor in layer_past:
                if tensor is not None:
                    repeated_tensor = tensor.repeat_interleave(repeats=repeats, dim=0)
                else:
                    repeated_tensor = None
                repeated_layer_past.append(repeated_tensor)
            repeated_past_key_values.append(tuple(repeated_layer_past))
        return tuple(repeated_past_key_values)


def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def evaluate_with_context(logits, y, args, test_at = [1, 63, 127], type=None):
    out_dict = {}
    bsz, d = logits.shape
    n_seq = int(bsz / (args.block_size))
    logits = logits.view(n_seq, args.block_size, d)
    
    if type != 'accuracy':
        y = y.view(n_seq, args.block_size, d)
    else:
        y = y.view(n_seq, args.block_size)
   
    # 
    for ctxt in test_at:
        if type == 'accuracy':
            acc = (logits[:, ctxt, :].argmax(dim=1) == y[:, ctxt]).float().mean()
            out_dict[f'acc_c={ctxt}'] = acc.item()
        else:
            mse = F.mse_loss(logits[:, ctxt, :], y[:, ctxt, :])
            r2 = r2_score(logits[:, ctxt, :], y[:, ctxt, :])
            out_dict[f'r2_c={ctxt}'] = r2.item()
    return out_dict


def evaluation_with_cache(model, online_linear_probe, cache, test_loader, test_mode, test_at = [0, 64, 128], args=None):
    cache_x, cache_r, cache_y = cache
    assert cache_x is not None
    cache_r_input = cache_r.clone()
    if test_mode == 'rot':
        cache_r_input[:, 4:] = 0
    elif test_mode == 'color':
        cache_r_input[:, :4] = 0
    elif test_mode == 'rotcolor':
        pass
    model.eval()
    online_linear_probe.eval()
    test_logs = {'Test acc feat': {'zr': {}, 'z+': {}}, 'Test acc pred': {'zr': {}, 'z+': {}}}
    test_logs.update({'Test Rot R2': {'zr': {}, 'z+': {}}, 'Test Color R2': {'zr': {}, 'z+': {}}})

    for ctxt_len in test_at:
        assert ctxt_len%2 ==0, 'So far only even context lengths are supported for evaluation'
        with torch.no_grad():  
            if ctxt_len == 0 :   
                initial_past = None
            else:
                curr_x = cache_x[:ctxt_len, :, :, :].cuda(non_blocking=True)
                curr_r = cache_r_input[:ctxt_len//2, :].cuda(non_blocking=True)
                initial_past = model(curr_x, curr_r, return_cache = True)[-1]
                initial_past = [[p[:, :, :ctxt_len, :] for p in layer] for layer in initial_past]
                initial_past = repeat_past_key_values(initial_past, args.batch_size)
                   
            loader_bar = tqdm(test_loader, desc=f'Eval ctxt={ctxt_len}')
            labels, latents, i_latents, inv_latents = [], [], [], []
            all_logits_zr_feat, all_logits_zplus_feat = [], []
            all_logits_zr_pred, all_logits_zplus_pred = [], []
            all_equiv_logits_zr, all_equiv_logits_zplus = [], []
            all_inv_logits_zr, all_inv_logits_zplus = [], []
            
            for x, xv, latent, i_latent, y in loader_bar:
                x, xv = x.cuda(non_blocking=True), xv.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                latent = latent.cuda(non_blocking=True)
                i_latent = i_latent.cuda(non_blocking=True)
                latent_input = latent.clone()
                
                if test_mode == 'rot':
                    latent_input[:, 4:] = 0
                    
                elif test_mode == 'color':
                    latent_input[:, :4] = 0
                    
                elif test_mode == 'rotcolor':
                    pass
                
                n_classes = 55

                # retrieving embeddings of 1) (z,r) ; 2) (z+, 0)
                feature, _, embd_zr, _ = model(x, latent_input, past = initial_past, return_cache = False, cache_inference_mode=True)
                feature, _, embd_zplus, _ = model(xv, torch.zeros_like(latent_input), past = initial_past, return_cache = False, cache_inference_mode=True)

                logits_zr_feat, logits_zr_pred =  online_linear_probe(feature, embd_zr)#.detach().cpu()
                logits_zplus_feat, logits_zplus_pred =  online_linear_probe(feature, embd_zplus)#.detach().cpu()
                logits_zr_feat, logits_zr_pred = logits_zr_feat.detach().cpu(), logits_zr_pred.detach().cpu()
                logits_zplus_feat, logits_zplus_pred = logits_zplus_feat.detach().cpu(), logits_zplus_pred.detach().cpu()
                
                if args.eval_normalize:
                    embd_zr = F.normalize(embd_zr, dim=1)
                    embd_zplus = F.normalize(embd_zplus, dim=1)
                equiv_logits_zr = model.aug_predictor(embd_zr).detach().cpu()
                equiv_logits_zplus = model.aug_predictor(embd_zplus).detach().cpu()
               
                all_logits_zr_feat.append(logits_zr_feat)
                all_logits_zr_pred.append(logits_zr_pred)
                all_logits_zplus_pred.append(logits_zplus_pred)
                all_logits_zplus_feat.append(logits_zplus_feat)
                all_equiv_logits_zr.append(equiv_logits_zr)
                all_equiv_logits_zplus.append(equiv_logits_zplus)
                
                
                i_latents.append(i_latent.detach().cpu())
                labels.append(y.detach().cpu())
                
            class_acc_zr_feat = (torch.cat(all_logits_zr_feat, dim=0).argmax(dim=1) == torch.cat(labels, dim=0)).float().mean()
            class_acc_zr_pred = (torch.cat(all_logits_zr_pred, dim=0).argmax(dim=1) == torch.cat(labels, dim=0)).float().mean()
            class_acc_zplus_feat = (torch.cat(all_logits_zplus_feat, dim=0).argmax(dim=1) == torch.cat(labels, dim=0)).float().mean()
            class_acc_zplus_pred = (torch.cat(all_logits_zplus_pred, dim=0).argmax(dim=1) == torch.cat(labels, dim=0)).float().mean()

            if test_mode == 'rot':
                rot_r2_zr = r2_score(torch.cat(all_equiv_logits_zr, dim=0)[:, :4], torch.cat(i_latents, dim=0)[:, 1, :4])
                rot_r2_zplus = r2_score(torch.cat(all_equiv_logits_zplus, dim=0)[:, :4], torch.cat(i_latents, dim=0)[:, 1, :4])
                color_r2_zr = r2_score(torch.cat(all_equiv_logits_zr, dim=0)[:, 4:], torch.cat(i_latents, dim=0)[:, 0, 4:])
                color_r2_zplus = r2_score(torch.cat(all_equiv_logits_zplus, dim=0)[:, 4:], torch.cat(i_latents, dim=0)[:, 1, 4:])
            elif test_mode == 'color':
                rot_r2_zr = r2_score(torch.cat(all_equiv_logits_zr, dim=0)[:, :4], torch.cat(i_latents, dim=0)[:, 0, :4])
                rot_r2_zplus = r2_score(torch.cat(all_equiv_logits_zplus, dim=0)[:, :4], torch.cat(i_latents, dim=0)[:, 1, :4])
                color_r2_zr = r2_score(torch.cat(all_equiv_logits_zr, dim=0)[:, 4:], torch.cat(i_latents, dim=0)[:, 1, 4:])
                color_r2_zplus = r2_score(torch.cat(all_equiv_logits_zplus, dim=0)[:, 4:], torch.cat(i_latents, dim=0)[:, 1, 4:])
            else:
                rot_r2_zr = r2_score(torch.cat(all_equiv_logits_zr, dim=0)[:, :4], torch.cat(i_latents, dim=0)[:, 1, :4])
                rot_r2_zplus = r2_score(torch.cat(all_equiv_logits_zplus, dim=0)[:, :4], torch.cat(i_latents, dim=0)[:, 1, :4])
                color_r2_zr = r2_score(torch.cat(all_equiv_logits_zr, dim=0)[:, 4:], torch.cat(i_latents, dim=0)[:, 1, 4:])
                color_r2_zplus = r2_score(torch.cat(all_equiv_logits_zplus, dim=0)[:, 4:], torch.cat(i_latents, dim=0)[:, 1, 4:])
                    
            test_logs['Test acc feat']['zr'][f'c={ctxt_len}'] = round(class_acc_zr_feat.item(), 4)
            test_logs['Test acc pred']['zr'][f'c={ctxt_len}'] = round(class_acc_zr_pred.item(), 4)
            test_logs['Test acc feat']['z+'][f'c={ctxt_len}'] = round(class_acc_zplus_feat.item(), 4)
            test_logs['Test acc pred']['z+'][f'c={ctxt_len}'] = round(class_acc_zplus_pred.item(), 4)
            test_logs['Test Rot R2']['zr'][f'c={ctxt_len}'] = round(rot_r2_zr.item(), 4)
            test_logs['Test Rot R2']['z+'][f'c={ctxt_len}'] = round(rot_r2_zplus.item(), 4)
            test_logs['Test Color R2']['zr'][f'c={ctxt_len}'] = round(color_r2_zr.item(), 4)
            test_logs['Test Color R2']['z+'][f'c={ctxt_len}'] = round(color_r2_zplus.item(), 4)

    return test_logs


