import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .position_encoding import PositionEmbeddingSine
from ...utils.utils import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn
from ..moe import MoFFN, ETCA
from .uagr import UAGR
from .hgc import HGC


def build_etca_encoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.DECODER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model
        

class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4,
                 attn_type="dst", ffn_type="ffn",
                 n_experts=4, k=1, n_points=4,
                 noisy_gating=True, acc_aux_loss=True,
                 # UAGR args
                 uagr_enabled=False, uagr_num_balls=32,
                 uagr_use_diag_cov=True, uagr_tau=1.0,
                 uagr_position="post_attn"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation,
            num_feature_levels, nhead,
            attn_type=attn_type, ffn_type=ffn_type,
            n_experts=n_experts, k=k,
            n_points=n_points,
            noisy_gating=noisy_gating, acc_aux_loss=acc_aux_loss,
            uagr_enabled=uagr_enabled,
            uagr_num_balls=uagr_num_balls,
            uagr_use_diag_cov=uagr_use_diag_cov,
            uagr_tau=uagr_tau,
            uagr_position=uagr_position,
        )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
        
    def forward(self, srcs, masks, pos_embeds):
        enable_mask = 0
        if masks is not None:
            for src in srcs:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory, losses_moe, loss_sampling_rank, uagr_intermediates, all_centers, all_features, all_attentions = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                                                                                                        lvl_pos_embed_flatten, mask_flatten)
        return memory, spatial_shapes, level_start_index, losses_moe, loss_sampling_rank, uagr_intermediates, all_centers, all_features, all_attentions


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8,
                 attn_type="dst", ffn_type="moffn",
                 n_experts=4, k=1, n_points=4,
                 noisy_gating=True, acc_aux_loss=True,
                 # uagr
                 uagr_enabled=False, uagr_num_balls=32,
                 uagr_use_diag_cov=True, uagr_tau=1.0,
                 uagr_position="post_attn"):
        super().__init__()
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        self.uagr_position = uagr_position if uagr_enabled else "none"

        # Attention
        if attn_type == "etca":
            self.self_attn = ETCA(n_experts=n_experts, k=k, d_model=d_model, n_levels=n_levels,
                                 n_heads=n_heads, n_points=n_points,
                                 noisy_gating=noisy_gating, acc_aux_loss=acc_aux_loss)
        else:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        if ffn_type == "moffn":
            self.ffn = MoFFN(n_experts=n_experts, k=k, d_model=d_model, d_ffn=d_ffn,
                             noisy_gating=noisy_gating, acc_aux_loss=acc_aux_loss, dropout=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ffn), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(d_ffn, d_model), nn.Dropout(dropout)
            )

        self.norm2 = nn.LayerNorm(d_model)

        # UAGR
        self.uagr = None
        if uagr_enabled:
            self.uagr = UAGR(d_model, num_balls=uagr_num_balls, use_diag_cov=uagr_use_diag_cov, tau=uagr_tau)
            if self.uagr_position == "parallel":
                self.uagr_alpha = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, _=0):
        loss_attn = {'topk_loss': torch.zeros((1,), device=src.device),
                     'switch_loss': torch.zeros((1,), device=src.device),
                     'z_loss': torch.zeros((1,), device=src.device)}
        loss_ffn = {'topk_loss': torch.zeros((1,), device=src.device),
                    'switch_loss': torch.zeros((1,), device=src.device),
                    'z_loss': torch.zeros((1,), device=src.device)}
        uagr_att = None
        uagr_dif = None
        uagr_sigma = None

        # Pre-Attn
        if self.uagr_position == "pre_attn":
            src_uagr, uagr_att, uagr_dif, uagr_sigma = self.uagr(src)
            src = src + src_uagr

        # # pre-etca
        # from PIL import Image
        # matrix = src[0, -12544:].reshape((112, 112, 256)).mean(dim=-1).float().cpu().numpy()
        # gray = (255 * (matrix - matrix.min()) / (matrix.max() - matrix.min())).astype(np.uint8)
        # gray = gray.repeat(10, 0).repeat(10, 1)
        # Image.fromarray(gray, 'L').save('adb/' + str(_)+ '-0-pre-etca.png')
        
        # Self Attention
        if self.attn_type == "dst":
            src2, loss_attn = self.self_attn(
                self.with_pos_embed(src, pos), reference_points, src,
                spatial_shapes, level_start_index, padding_mask
            )
        if self.attn_type == "etca":
            src2, loss_attn = self.self_attn(
                self.with_pos_embed(src, pos), reference_points, src,
                spatial_shapes, level_start_index, padding_mask
            )
        else:
            src2 = self.self_attn(
                self.with_pos_embed(src, pos), reference_points, src,
                spatial_shapes, level_start_index, padding_mask
            )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # from PIL import Image
        # matrix = src[0, -12544:].reshape((112, 112, 256)).mean(dim=-1).float().cpu().numpy()
        # gray = (255 * (matrix - matrix.min()) / (matrix.max() - matrix.min())).astype(np.uint8)
        # gray = gray.repeat(10, 0).repeat(10, 1)
        # Image.fromarray(gray, 'L').save('adb/' + str(_)+ '-1-post-etca.png')

        # Post-Attn
        if self.uagr_position == "post_attn":
            src_uagr, uagr_att, uagr_dif, uagr_sigma = self.uagr(src)
            src = src + src_uagr

        # FFN
        if self.ffn_type == "moffn":
            src3, loss_ffn = self.ffn(src)
        else:
            src3 = self.ffn(src)

        # Parallel
        if self.uagr_position == "parallel":
            src_uagr, uagr_att, uagr_dif, uagr_sigma = self.uagr(src)
            src = self.norm2(src + self.uagr_alpha * src3 + (1 - self.uagr_alpha) * src_uagr)
        else:
            src = self.norm2(src + src3)

        # # Post-FFN
        # from PIL import Image
        # matrix = src[0, -12544:].reshape((112, 112, 256)).mean(dim=-1).float().cpu().numpy()
        # gray = (255 * (matrix - matrix.min()) / (matrix.max() - matrix.min())).astype(np.uint8)
        # gray = gray.repeat(10, 0).repeat(10, 1)
        # Image.fromarray(gray, 'L').save('adb/' + str(_)+ '-2-post-sgfe.png')

        if self.uagr_position == "post_ffn":
            src_uagr, uagr_att, uagr_dif, uagr_sigma = self.uagr(src)
            src = src + src_uagr

        # # Post-UAGR
        # from PIL import Image
        # matrix = src[0, -12544:].reshape((112, 112, 256)).mean(dim=-1).float().cpu().numpy()
        # gray = (255 * (matrix - matrix.min()) / (matrix.max() - matrix.min())).astype(np.uint8)
        # gray = gray.repeat(10, 0).repeat(10, 1)
        # Image.fromarray(gray, 'L').save('adb/' + str(_)+ '-3-post-uagr.png')

        return src, loss_attn, loss_ffn, uagr_att, uagr_dif, uagr_sigma


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        losses_moe = {
            'topk_loss': torch.zeros((1,), device=src.device),
            'switch_loss': torch.zeros((1,), device=src.device),
            'z_loss': torch.zeros((1,), device=src.device)
        }
        loss_sampling_rank = {'loss_sampling_rank': torch.zeros((1,), device=src.device),}
        uagr_intermediates = []

        all_centers = []
        all_features = []
        all_attentions = []

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output, loss_attn, loss_ffn, att, dif, sigma = layer(
                output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, _
            )
            for k in losses_moe.keys():
                losses_moe[k] += loss_attn[k]
                losses_moe[k] += loss_ffn[k]
            for k in loss_sampling_rank.keys():
                loss_sampling_rank[k] += loss_attn[k]
            if att is not None:
                uagr_intermediates.append({"att": att, "dif": dif, "sigma": sigma})
                all_centers.append(layer.uagr.centers)      # (K, D)
                all_features.append(output)                # (B, N, D)
                all_attentions.append(att)                 # (B, N, K)
        return output, losses_moe, loss_sampling_rank, uagr_intermediates, all_centers, all_features, all_attentions


@SEM_SEG_HEADS_REGISTRY.register()
class ETCA_Encoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        transformer_in_features: List[str],
        common_stride: int,
        num_feature_levels: int,
        total_num_feature_levels: int,
        feature_order: str,
        n_points: int,
        attn_type: str,
        ffn_type: str,
        n_experts: int,
        k: int,
        noisy_gating: bool,
        acc_aux_loss: bool,
        # UAGR
        uagr_enabled: bool = False,
        uagr_num_balls: int = 32,
        uagr_use_diag_cov: bool = True,
        uagr_tau: float = 1.0,
        uagr_position: str = "post_attn",
    ):
        super().__init__()
        
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }
        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        self.feature_order = feature_order

        if feature_order == "low2high":
            transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: -x[1].stride)
        else:
            transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.dst_num_feature_levels = num_feature_levels  # always use 3 scales
        self.total_num_feature_levels = total_num_feature_levels
        self.common_stride = common_stride

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        self.low_resolution_index = transformer_in_channels.index(max(transformer_in_channels))
        self.high_resolution_index = 0 if self.feature_order == 'low2high' else -1
        if self.transformer_num_feature_levels > 1:
            # input projection layers for different levels
            input_proj_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            # input projectino for downsample [more levels not output by backbone]
            in_channels = max(transformer_in_channels)
            for _ in range(self.total_num_feature_levels - self.transformer_num_feature_levels):  # exclude the res2
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, conv_dim),
                ))
                in_channels = conv_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            # input projection for single level
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])
        # init input projection layers
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.total_num_feature_levels,
            attn_type=attn_type,
            ffn_type=ffn_type,
            n_experts=n_experts,
            k=k,
            n_points=n_points,
            noisy_gating=noisy_gating,
            acc_aux_loss=acc_aux_loss,
            # UAGR
            uagr_enabled=uagr_enabled,
            uagr_num_balls=uagr_num_balls,
            uagr_use_diag_cov=uagr_use_diag_cov,
            uagr_tau=uagr_tau,
            uagr_position=uagr_position,
        )

        # positional encoding
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = max(int(np.log2(stride) - np.log2(self.common_stride)), 1)

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.DECODER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.DECODER.NHEADS
        ret["transformer_dim_feedforward"] = cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD  # deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES  # ['res3', 'res4', 'res5']
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        ret["total_num_feature_levels"] = cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS
        ret["num_feature_levels"] = cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS
        ret["feature_order"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER
        ret["n_points"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS
        ret["attn_type"] = cfg.MODEL.MOE.ATTN_TYPE_ENC
        ret["ffn_type"] = cfg.MODEL.MOE.FFN_TYPE_ENC
        ret["n_experts"] = cfg.MODEL.MOE.NUM_EXPERTS
        ret["k"] = cfg.MODEL.MOE.K
        ret["n_points"] = cfg.MODEL.MOE.N_POINTS
        ret["noisy_gating"] = cfg.MODEL.MOE.NOISY_GATING
        ret["acc_aux_loss"] = cfg.MODEL.MOE.ACC_AUX_LOSS

        # UAGR
        ret["uagr_enabled"] = cfg.MODEL.UAGR.ENABLED
        ret["uagr_num_balls"] = cfg.MODEL.UAGR.NUM_BALLS
        ret["uagr_use_diag_cov"] = cfg.MODEL.UAGR.USE_DIAG_COV
        ret["uagr_tau"] = cfg.MODEL.UAGR.TAU
        ret["uagr_position"] = cfg.MODEL.UAGR.POSITION
        return ret

    @autocast(enabled=False)
    def forward_features(self, features, masks):
        """
        :param features: multi-scale features from the backbone
        :param masks: image mask
        :return: enhanced multi-scale features and mask feature (1/4 resolution) for the decoder to produce binary mask
        """
        # backbone features
        srcs = []  # features that have been projected to transformer dim
        pos = []   # positional encoding
        # additional downsampled features (more scales not provided by backbone)
        srcsl = []
        posl = []
        # generate more features if there are more scales
        if self.total_num_feature_levels > self.transformer_num_feature_levels:
            # get feature maps with the lowest resolution
            smallest_feat = features[self.transformer_in_features[self.low_resolution_index]].float()
            _len_srcs = self.transformer_num_feature_levels
            for l in range(_len_srcs, self.total_num_feature_levels):
                # 3x3 kernel, stride 2
                if l == _len_srcs:
                    src = self.input_proj[l](smallest_feat)
                else:
                    src = self.input_proj[l](srcsl[-1])
                srcsl.append(src)
                posl.append(self.pe_layer(src))
        srcsl = srcsl[::-1]
        # Reverse feature maps
        # process backbone features
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
        # high2low: additional-backbone; low2high: backbone-additional
        srcs.extend(srcsl) if self.feature_order == 'low2high' else srcsl.extend(srcs)
        pos.extend(posl) if self.feature_order == 'low2high' else posl.extend(pos)
        if self.feature_order != 'low2high':
            srcs = srcsl
            pos = posl
            
        y, spatial_shapes, level_start_index, losses_moe, loss_sampling_rank, uagr_intermediates, all_centers, all_features, all_attentions = self.transformer(
            srcs, masks, pos
        )
        bs = y.shape[0]

        split_size_or_sections = [None] * self.total_num_feature_levels
        for i in range(self.total_num_feature_levels):
            if i < self.total_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            y = cur_fpn + F.interpolate(
                out[self.high_resolution_index], size=cur_fpn.shape[-2:],
                mode="bilinear", align_corners=False
            )
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.total_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), out[0], multi_scale_features, losses_moe, loss_sampling_rank, uagr_intermediates, all_centers, all_features, all_attentions
