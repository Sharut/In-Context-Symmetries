"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional, Tuple

def get_pair_attention_mask(context_len, causal=True):
    if causal:
        mask = torch.ones(context_len, context_len).tril()
    else:
        mask = torch.ones(context_len, context_len)
    for i in range(context_len):
        if i % 2 == 1:
            mask[i, i-1] = 0
    mask = mask.bool()
    return mask


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).cuda()
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
            
        self.ctxt_len = config.block_size
        self.config = config

    def forward(self, x, layer_past=None, shuffle=False, omask=None, random_mask=False, mask_prob=0.15):
        B, T, C = x.size()
        
        # Split the linear projection into query, key, and values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Shape the query, key, and value tensors
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Concatenate past key and value tensors if they exist
        if layer_past is not None:
            past_key, past_value = layer_past
            past_size = past_key.shape[-2]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        # Update layer_past for the next usage
        present = (k, v)
        
        if self.flash:
            if layer_past is None:
                attnmask = get_pair_attention_mask(k.shape[-2]).cuda()
                if random_mask:
                    # mask with group 2
                    assert len(attnmask.shape) == 2
                    rdm = torch.rand(attnmask.shape[0], attnmask.shape[1]//2).cuda()
                    pair_mask = (rdm >= mask_prob).repeat_interleave(2, dim=1)
                    # set diagonal to 1
                    pair_mask = pair_mask | torch.eye(attnmask.shape[0], attnmask.shape[1]).bool().cuda()
                    attnmask = attnmask & pair_mask
            else:
                assert T==1, "Flash attention only supports T=1 for now"
                # One can also set attention mask to None here
                attnmask = torch.ones(T, k.shape[-2]).cuda()
                # attnmask = None

            y = scaled_dot_product_attention(q, k, v, attn_mask=attnmask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            raise ValueError('Not supported')

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, present  # Return both output and updated key/value pairs


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past=None, shuffle=False, omask=None, random_mask=False, mask_prob=0.15):
        out, present = self.attn(self.ln_1(x), layer_past = past, shuffle=shuffle, omask=omask, random_mask=random_mask, mask_prob=mask_prob)
        x = x + out
        x = x + self.mlp(self.ln_2(x))
        return x, present

# @@dataclass
class GPTConfig:
    block_size: int = 50
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    projection_dim: int = 128
    dropout: float = 0.0
    bias: bool = True
    aug_method: str = 'concat'
    is_invariant: bool =  False
    add_scale: float = 0.1
    transform_augs: bool = False

    def __init__(self, config_dict=None):
        if config_dict is not None:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                # Ignore extra elements not in GPTConfig
                else:
                    pass

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        r_dim = 6                                           # Set r_dim to the total dimension of augmentation vector
        
        if self.config.is_invariant:
            readin_dim = config.projection_dim
        elif self.config.aug_method == 'concat':            
            readin_dim = config.projection_dim + r_dim
        else:
            raise NotImplementedError
        
        if self.config.transform_augs:
            readin_dim = 2 * config.projection_dim
        
        self._read_in = nn.Linear(readin_dim, config.n_embd)
        self._read_out = nn.Linear(config.n_embd, config.projection_dim)
        self.context_len = config.block_size

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("=> Number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _combine(self, inp, augs):
        if augs.ndim == 1:
            augs = augs.unsqueeze(-1)
        augs = augs.cuda()
        
        # Prepare input tensor
        bsz, d = inp.shape
        z1, z2 = inp.view(bsz // 2, 2, d).unbind(dim=1) 

        if self.config.aug_method == 'concat':
            dummy = torch.zeros(bsz // 2, augs.shape[-1], device=z2.device)
            z1 = torch.cat([z1, augs], dim=-1)
            z2_wide = torch.cat((z2, dummy,), axis=1)
            sizes = z1.size()
            z = torch.stack((z1, z2_wide)).permute(1,0,2).reshape(sizes[0]*2, sizes[1]).float()
        
        return z, z1, z2_wide
        

    def forward(self, inp, augs, past_key_values=None, inference=False, random_mask=False, mask_prob=0.15):
        if past_key_values is None:
            past_key_values = [None] * self.config.n_layer

        if inference: # only used for evaluate_with_cache
            if not self.config.is_invariant:
                inp = torch.cat([inp, augs], dim=1)
            inp = inp.unsqueeze(1)
        else:
            
            bsz, d = inp.shape
            n_seq = bsz // self.config.block_size
            dummy_size = n_seq * self.config.block_size
            if bsz - dummy_size > 0:
                inp = torch.cat((inp, torch.zeros(dummy_size + self.config.block_size - bsz, d).cuda()), dim=0)
                augs = torch.cat((augs, torch.zeros((dummy_size + self.config.block_size - bsz)//2, augs.shape[-1]).cuda()), dim=0)
                n_seq += 1
            if not self.config.is_invariant:
                inp, inp1, inp2 = self._combine(inp, augs)  
            else:
                bsz, d = inp.shape
                inp1, inp2 = inp.view(bsz // 2, 2, d).unbind(dim=1) 
            bsz, d = inp.shape
            inp = inp.view(n_seq, self.config.block_size, d)
            inp1 = inp1.view(n_seq, self.config.block_size//2, d)
            inp2 = inp2.view(n_seq, self.config.block_size//2, d)
        
        inputs_embeds = self._read_in(inp)
        b, t = inputs_embeds.shape[0], inputs_embeds.shape[1]
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # transformer block processing
        x = self.transformer.drop(inputs_embeds)
        presents = []
        for i, (block, past) in enumerate(zip(self.transformer.h, past_key_values)):
            x, present = block(x, past, random_mask=random_mask, mask_prob=mask_prob)
            presents.append(present)
    
        x = self.transformer.ln_f(x)
        x = self._read_out(x)

        # If drop_last is False, adjusting for batch size mismatch of the last batch
        if not inference:
            if bsz - dummy_size > 0:
                x = x.view(dummy_size + self.config.block_size, -1)
                x = x[:bsz, :]
            else:
                x = x.view(bsz, -1)
        else:
            x = x.squeeze(1)
        return x, presents

        