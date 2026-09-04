# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
from ..functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from ...moe import ParallelExperts


@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    # probs: original probs
    # top_k_gates: logits of topK gating function
    # top_k_indices: indices of topK exprets
    
    # create an array containing the topk logits only
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    
    # flat for sorting
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    
    # exclude zero elements and sort the remaining
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    
    # count
    # number of selected experts
    expert_size = (gates > 0).long().sum(0)
    # indices of selected and sorted experts
    index_sorted_experts = nonzeros[_index_sorted_experts]
    # map sorted indices to original indices
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    # get corrssponding gating logits (sorted)
    batch_gates = top_k_gates[index_sorted_experts]
    
    return batch_gates, batch_index, expert_size, gates, index_sorted_experts


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MoS(nn.Module):
    def __init__(self, n_experts=4, k=1, d_model=256, n_levels=4, n_heads=8, n_points=[1, 2, 4, 8, 16, 32, 64, 128], noisy_gating=True, w_topk_loss=0.001, w_switch_loss=0.001,
                 w_z_loss=0.001):
        """
        Mixture-of-Sampling (MoS) Module (based on Multi-Scale Deformable Attention Module)
        Different experts own different sampling points
        
        :param n_experts    number of experts (for both offset layer and attention layer)
        :param k            number of selected expert, k must be 1 to avoid to modify the cuda implemtntation of MSDA
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level in an expert
        :param noisy_gating whether add noise to clear logits of gate
        :oaram w_topk_loss  weight of topk loss
        :param w_switch_loss weight of switch loss
        :param w_z_loss     weight of z_loss
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        assert n_experts >=1 and n_experts <= len(n_points)
        assert k == 1
        
        self.im2col_step = 128

        self.n_experts = n_experts
        self.k = k
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        max_points = n_points[n_experts-1]
        self.max_points = max_points
        # experts with different output shapes cannot be conducted in parallel, so we apply mask to activate corresponding points
        # [1, 0, 0, 0, 0, 0, 0, 0]
        # [1, 1, 0, 0, 0, 0, 0, 0]
        # [1, 1, 1, 1, 0, 0, 0, 0]
        # [1, 1, 1, 1, 1, 1, 1, 1]  this is an example of 4 experts with [1, 2, 4, 8] sampling points
        mask_template = torch.zeros((n_experts, max_points))           # (n_experts, max_points)
        for i in range(n_experts):
            mask_template[i, :n_points[i]] = 1 
        mask_template = mask_template.unsqueeze(1)                      # (n_experts, 1, max_points))
        self.mask_template = mask_template.repeat((1, n_levels, 1))     # (n_experts, 4, max_points)

        # gate layer
        self.gate = nn.Linear(d_model, 2 * n_experts if noisy_gating else n_experts, bias=False)

        # expert layers
        self.sampling_experts = ParallelExperts(n_experts, d_model, n_heads * n_levels * max_points * 2, bias=True)
        self.attention_experts = ParallelExperts(n_experts, d_model, n_heads * n_levels * max_points, bias=True)

        # loss
        self.w_topk_loss = w_topk_loss
        self.w_switch_loss = w_switch_loss
        self.w_z_loss = w_z_loss
        
        # regular layers
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()
        self.init_aux_statistics()

    def _reset_parameters(self):
        constant_(self.gate.weight.data, 0.)
        
        constant_(self.sampling_experts.w.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(1, self.n_heads, 1, 1, 2).repeat(self.n_experts, 1, self.n_levels, self.max_points, 1)
        for i in range(self.max_points):
            grid_init[:, :, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_experts.b = nn.Parameter(grid_init.view(-1))
        
        constant_(self.attention_experts.w.data, 0.)
        constant_(self.attention_experts.b.data, 0.)
        
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def init_aux_statistics(self, clear=True):
        # initialize the statistics of probality, topk probality, frequency of router
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.

        if clear:
            self.topk_acc_probs = 0.
    
    def update_aux_statistics(self, logits, probs, gates):
        # update existing statistics of router
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.0001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)
        
        self.topk_acc_probs = self.topk_acc_probs + probs.mean(0)

    def get_topk_loss_and_clear(self):
        # select topk probs and corresponding indices
        top_k_probs, top_k_indices = self.topk_acc_probs.topk(self.k, dim=0)
        # create the array with the same shape containing the topk probs only
        zeros = torch.zeros_like(self.topk_acc_probs)
        gates = zeros.scatter(0, top_k_indices, top_k_probs)
        # squre of topk probs and original probs (MSE)
        topk_loss = ((self.topk_acc_probs - gates) * (self.topk_acc_probs - gates)).sum()
        
        # reset topk_acc_probs
        self.topk_acc_probs = 0.
        return topk_loss * self.w_topk_loss 

    def get_aux_loss_and_clear(self):
        '''
            acc_gates: sum of topk soft score
            acc_freq: the number of being chosen
            acc_probs: sum of probs (probs = softmax(score))
        '''
        # compute losses
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_mlp_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)
        # weighted
        loss = self.switchloss * switchloss + self.zloss * zloss
        
        self.init_aux_statistics(clear=False)
        return loss
        
    def compute_switchloss(self, probs, freqs):
        # load-banlance loss
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.n_experts
        
    def compute_zloss(self, logits):
        # mean(log(sum(e(x)))^2)
        # suppress the maximum logit
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss
    
    def top_k_gating(self, x, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # get original gating logits
        clean_logits = self.gate(x)
        # noisy gating during training
        if self.noisy_gating and self.training:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        # noisy gating during test
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        # no noisy gating
        else:
            logits = clean_logits
        
        # activate original logits
        probs = torch.softmax(logits, dim=1) + 1e-4
        
        # top-1
        top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        
        # sort selected logits and get corresponding indices
        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates
        
        # compute losses
        loss = 0.
        self.update_aux_statistics(logits, probs, gates)
        loss += self.switchloss * self.compute_switchloss(probs, self.expert_size)
        loss += self.zloss * self.compute_zloss(logits)
        loss = torch.Tensor([[[loss,]]])
        return loss
    
    def keep_valid_points(self, x, sampling_):
        '''
        for both sampling experts and attention experts
        '''
        pass
    
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, sum_{l=0}^{L-1} H_l cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, sum_{l=0}^{L-1} H_l cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # gating
        loss = self.top_k_gating(query)
        
        # routing
        expert_input = x[self.batch_index]
        # experts
        sampling_offsets = self.sampling_experts(expert_input, self.expert_size).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_experts(expert_input, self.expert_size).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # mask operation
        mask4sampling = self.mask_template.reshape((1, 1, 1, self.n_levels, self.n_points, 1))
        mask4sampling = mask4sampling.repeat((N, Len_q, self.n_heads, 1, 1, 2))
        sampling_offsets = sampling_offsets * mask4sampling
        mask4attention = self.mask_template.reshape((1, 1, 1, self.n_levels*self.n_points)) # [1 0 0 0]
        mask4attention = mask4attention.repeat((N, Len_q, self.n_heads, 1))                 # [1 0 0 0]
        mask4attention[mask4attention == 0] = -1e6                                          # [1 -1e6 -1e6 -1e6]
        attention_weights = attention_weights * mask4attention
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        # multiply original logits
        sampling_offsets = sampling_offsets * self.batch_gates[:, None]
        attention_weights = attention_weights * self.batch_gates[:, None]
        
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        try:
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        except:
            # CPU
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        # # For FLOPs calculation only
        # output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        # copy the outputs of selected experts to all experts
        # zeros = torch.zeros((bsz * length, emb_size), 
        #     dtype=expert_outputs.dtype, device=expert_outputs.device)
        # y = zeros.index_add(0, self.batch_index, expert_outputs)
        # y = y.view(bsz, length, emb_size)
        
        return output, loss





