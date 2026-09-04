# Mixture-of-Sampling (MoS) Module

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..pixel_decoder.ops.functions import MSDeformAttnFunction
from ..pixel_decoder.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from .base.parallel_experts import ParallelExperts


class EdgeAwareProj(nn.Module):
    """Edge perception using 1d conv"""
    def __init__(self, dim):
        super().__init__()
        # differential operator
        self.diff_conv = nn.Conv1d(dim, dim//4, kernel_size=3, padding=1, groups=dim//4)
        with torch.no_grad():
            # initialize kernel [−0.5, 0, 0.5]
            self.diff_conv.weight[:] = 0
            self.diff_conv.weight[:, 0, 0] = -0.5
            self.diff_conv.weight[:, 0, 2] = 0.5
        self.norm = nn.LayerNorm(dim//4)
    
    def forward(self, x):
        x = x.unsqueeze(-1)  # (B*N, D, 1)
        x = self.diff_conv(x)  # (B*N, D/4, 1)
        x = x.permute(0, 2, 1).squeeze(1)  # (B*N, D/4)
        return self.norm(x)

class RegionAwareProj(nn.Module):
    """Region perception using average pooling"""
    def __init__(self, dim):
        super().__init__()
        self.smooth = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Linear(dim, dim//4)
        self.norm = nn.LayerNorm(dim//4)
    
    def forward(self, x):
        x = x.unsqueeze(-1).permute(0, 2, 1)  # (B*N, 1, D)
        x = self.smooth(x)
        x = x.permute(0, 2, 1).squeeze(-1)
        return self.norm(self.proj(x))

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


class ETCA(nn.Module):
    def __init__(self, n_experts=4, k=1, d_model=256, n_levels=4, n_heads=8, n_points=4, noisy_gating=True, acc_aux_loss=True):
        """
        Mixture-of-Collaborative-Attention (ETCA)
        Dual Deformable Sampling for Edge and Region
        
        :param n_experts    number of experts (for both offset layer and attention layer)
        :param k            number of selected expert, k must be 1 to avoid to modify the cuda implemtntation of MSDA
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points using fixed pattern
        :param noisy_gating whether add noise to clear logits of gate
        :param acc_aux_loss save the usage of experts
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        assert n_experts >=1
        
        self.im2col_step = 128

        self.n_experts = n_experts
        self.k = k
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points     
        self.noisy_gating = noisy_gating
        self.acc_aux_loss = acc_aux_loss
        
        # gate layer
        self.gate = nn.Linear(d_model, 2 * n_experts if noisy_gating else n_experts, bias=False)
        self.collaboration_gate =  nn.Sequential(
                                                nn.Linear(d_model // 4 * 2, d_model // 4),
                                                nn.ReLU(),
                                                nn.Linear(d_model // 4, 2),
                                                nn.Softmax(dim=-1)
                                            )

        # Dual Sampling Experts
        self.dual_sampling_experts = ParallelExperts(n_experts, d_model, n_heads * n_levels * n_points * 2 * 2, bias=True)
        self.dual_attention_experts = ParallelExperts(n_experts, d_model, n_heads * n_levels * n_points * 2, bias=True)
        
        # regular layers
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # perception layers
        self.edge_proj = EdgeAwareProj(d_model)
        self.region_proj = RegionAwareProj(d_model)

        # rank loss
        self.rank_loss_weight = 0.1

        self._reset_parameters()
        self.init_aux_statistics()
        # expert frequency
        self.expert_frequency = torch.zeros((n_experts,), device='cuda')

    def _reset_parameters(self):
        constant_(self.gate.weight.data, 0.)
        
        constant_(self.dual_sampling_experts.w.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(1, self.n_heads, 1, 1, 2, 1).repeat(self.n_experts, 1, self.n_levels, self.n_points, 1, 2)
        for i in range(self.n_points):
            grid_init[:, :, :, i, :, :] *= i + 1
        with torch.no_grad():
            self.dual_sampling_experts.b = nn.Parameter(grid_init.view((self.n_experts, -1)))
        
        constant_(self.dual_attention_experts.w.data, 0.)
        constant_(self.dual_attention_experts.b.data, 0.)
        
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
        return {'topk_loss': topk_loss}
    
    def get_aux_loss_and_clear(self):
        '''
            acc_gates: sum of topk soft score
            acc_freq: the number of being chosen
            acc_probs: sum of probs (probs = softmax(score))
        '''
        # compute losses
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.n_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)
        zloss = torch.where(
            torch.isinf(zloss) | torch.isnan(zloss),
            torch.tensor(1e6, device=zloss.device),
            zloss
        )
        
        # print expert frequency
        self.expert_frequency += self.acc_freq
        # print(id(self.expert_frequency), self.expert_frequency)
        
        self.init_aux_statistics(clear=False)
        return {'switch_loss': switchloss, 'z_loss': zloss}
        
    def compute_switchloss(self, probs, freqs):
        # load-banlance loss
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.n_experts
        
    def compute_zloss(self, logits):
        # mean(log(sum(e(x)))^2)
        # suppress the maximum logit
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        zloss = torch.where(
            torch.isinf(zloss) | torch.isnan(zloss),
            torch.tensor(1e6, device=zloss.device),
            zloss
        )
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
        if self.acc_aux_loss:
            self.update_aux_statistics(logits, probs, gates)
        else:
            loss += self.switchloss * self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
        loss = torch.Tensor([[[loss,]]])
        return loss, logits, probs, gates
    
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

        # each query is the basic unit for an expert, there are N*Len_q queries
        query = query.reshape((N*Len_q, self.d_model))
        
        # gating
        loss, logits, probs, gates = self.top_k_gating(query)
        
        # routing
        # Original Index ==> Expert Index
        expert_input = query[self.batch_index]

        # ==================================================================================================
        # Sampling Experts (shared gate)
        dual_sampling_offsets = self.dual_sampling_experts(expert_input, self.expert_size)
        dual_sampling_offsets = dual_sampling_offsets * self.batch_gates[:, None]
        dual_sampling_offsets = dual_sampling_offsets.view(N*Len_q*self.k, self.n_heads, self.n_levels, self.n_points, 2, 2)

        offset_edge = dual_sampling_offsets[..., 0, :]
        offset_region = dual_sampling_offsets[..., 1, :]
        rank_loss = self.sampling_ranking_loss(offset_edge, offset_region)
        
        # Expert Index ==> Original Index
        # transform indexes of patches to Original Space
        zeros = torch.zeros((N*Len_q, self.n_heads, self.n_levels, self.n_points, 2), dtype=dual_sampling_offsets.dtype, device=dual_sampling_offsets.device)
        offset_edge = zeros.index_add(0, self.batch_index, offset_edge)
        offset_edge = offset_edge.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        zeros = torch.zeros((N*Len_q, self.n_heads, self.n_levels, self.n_points, 2), dtype=dual_sampling_offsets.dtype, device=dual_sampling_offsets.device)
        offset_region = zeros.index_add(0, self.batch_index, offset_region)
        offset_region = offset_region.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
      
        # ==================================================================================================
        # Attention Experts (shared gate)
        dual_attention_weights = self.dual_attention_experts(expert_input, self.expert_size)
        dual_attention_weights = dual_attention_weights * self.batch_gates[:, None]
        dual_attention_weights = dual_attention_weights.view(N*Len_q*self.k, self.n_heads, self.n_levels, self.n_points, 2)
        
        attn_edge = dual_attention_weights[..., 0].view(N*Len_q*self.k, self.n_heads, self.n_levels * self.n_points)
        attn_region = dual_attention_weights[..., 1].view(N*Len_q*self.k, self.n_heads, self.n_levels * self.n_points)
        
        attn_edge = F.softmax(attn_edge, -1).view(N*Len_q*self.k, self.n_heads, self.n_levels, self.n_points)
        attn_region = F.softmax(attn_region, -1).view(N*Len_q*self.k, self.n_heads, self.n_levels, self.n_points)
        # Expert Index ==> Original Index
        zeros = torch.zeros((N*Len_q, self.n_heads, self.n_levels, self.n_points), dtype=dual_attention_weights.dtype, device=dual_attention_weights.device)
        attn_edge = zeros.index_add(0, self.batch_index, attn_edge)
        attn_edge = attn_edge.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        zeros = torch.zeros((N*Len_q, self.n_heads, self.n_levels, self.n_points), dtype=dual_attention_weights.dtype, device=dual_attention_weights.device)
        attn_region = zeros.index_add(0, self.batch_index, attn_region)
        attn_region = attn_region.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        
        # ==================================================================================================
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations_edge = reference_points[:, :, None, :, None, :] \
                                         + offset_edge / offset_normalizer[None, None, None, :, None, :]
            sampling_locations_region = reference_points[:, :, None, :, None, :] \
                                         + offset_region / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations_edge = reference_points[:, :, None, :, None, :2] \
                                         + offset_edge / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            sampling_locations_region = reference_points[:, :, None, :, None, :2] \
                                         + offset_region / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
            
        try:
            output_edge = MSDeformAttnFunction.apply(
                    value, input_spatial_shapes, input_level_start_index, sampling_locations_edge, attn_edge, self.im2col_step)
        except:
            # CPU
            output_edge = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations_edge, attn_edge)
        try:
            output_region = MSDeformAttnFunction.apply(
                    value, input_spatial_shapes, input_level_start_index, sampling_locations_region, attn_region, self.im2col_step)
        except:
            # CPU
            output_region = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations_region, attn_region)
        # # For FLOPs calculation only
        # output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)

        # Collaboration gate
        gate_input_per_query = torch.cat([
            self.edge_proj(query), self.region_proj(query)
        ], dim=-1)  # (N*Len_q, D/2)
        comp_gate = self.collaboration_gate(gate_input_per_query).view(N, Len_q, 2)  # (N*Len_q, 2)   
        output_edge = output_edge * comp_gate[..., 0:1]  # broadcast
        output_region = output_region * comp_gate[..., 1:2]
        fused_output = output_edge + output_region
        
        output = self.output_proj(fused_output)

        losses = {}
        losses.update(self.get_topk_loss_and_clear())
        losses.update(self.get_aux_loss_and_clear())
        losses['loss_sampling_rank'] = rank_loss

        return output, losses

    # def sampling_ranking_loss(self, offsets_a, offsets_b):
    #     """Constraint: average distance from edge sampling points more than that from region sampling points"""
    #     dist_a = offsets_a.norm(dim=-1).mean()
    #     dist_b = offsets_b.norm(dim=-1).mean()
    #     return torch.clamp(dist_b - dist_a, min=0)

    def sampling_ranking_loss(self, offsets_a, offsets_b, lambda_reg=0.05):
        """
        Sampling Rank Loss for ETCA, it significant reduce computational complexity compared with Chamfer distance.
        
        Args:
            offsets_a: [N, H, L, P, 2], edge sampling offsets (larger range)
            offsets_b: [N, H, L, P, 2], region sampling offsets (inner range)
            lambda_reg: weight for compactness term (default 0.05)
        """
        # 1. 径向距离 [N, H, L, P]
        r_a = offsets_a.norm(dim=-1)
        r_b = offsets_b.norm(dim=-1)
    
        # 2. 方向硬匹配（无 einsum，无 softmax）
        # 对 P=4 逐点计算方向一致性分数（广播点积），避免构造稠密矩阵
        # offsets_a[..., k:k+1, :] 为 [N,H,L,1,2]，与 offsets_b [N,H,L,P,2] 广播相乘
        s0 = (offsets_b * offsets_a[..., 0:1, :]).sum(dim=-1)  # [N,H,L,P]
        s1 = (offsets_b * offsets_a[..., 1:2, :]).sum(dim=-1)
        s2 = (offsets_b * offsets_a[..., 2:3, :]).sum(dim=-1)
        s3 = (offsets_b * offsets_a[..., 3:4, :]).sum(dim=-1)
    
        # 堆叠后取最大方向一致性，得到最近邻索引 [N,H,L,P]
        scores = torch.stack([s0, s1, s2, s3], dim=-1)   # [N,H,L,P,4]
        _, nn_idx = scores.max(dim=-1)                     # [N,H,L,P]
    
        # 3. 覆盖损失：同方向上的 edge 半径必须大于 region
        r_a_matched = torch.gather(r_a, dim=-1, index=nn_idx)  # [N,H,L,P]
        loss_cover = torch.clamp(r_b - r_a_matched, min=0).mean()
    
        # 4. 轻量紧致正则：约束两组采样点的几何中心接近
        # 防止 edge 点为了“半径大”而整体偏移到远离 region 的方向
        loss_compact = (offsets_b - offsets_a).norm(dim=-1).mean()
    
        return loss_cover + lambda_reg * loss_compact + torch.tensor(0.5, device=offsets_a.device)



        