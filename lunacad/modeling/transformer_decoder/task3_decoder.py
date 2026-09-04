# MOBA Decoder

import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry
from detectron2.structures import BitMasks

from ...utils.utils import MLP, gen_encoder_output_proposals, inverse_sigmoid
from ...utils import box_ops


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskDINO.
"""


def build_task3_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.DECODER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


@TRANSFORMER_DECODER_REGISTRY.register()
class Task3_Decoder(nn.Module):
    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            mask_dim: int,
            enforce_input_project: bool,
            total_num_feature_levels: int = 4,
            semantic_ce_loss: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            enc_layers: number of Transformer encoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            d_model: transformer dimension
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels

        # define Transformer decoder here
        self.total_num_feature_levels = total_num_feature_levels

        self.num_queries = num_queries
        self.semantic_ce_loss = semantic_ce_loss
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes=num_classes
        # output FFNs
        assert self.mask_classification, "why not class embedding?"
        if self.mask_classification:
            if self.semantic_ce_loss:
                self.class_embed = nn.Linear(hidden_dim, num_classes+1)
            else:
                self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.label_enc=nn.Embedding(num_classes,hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)


    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.DECODER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.DECODER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["enforce_input_project"] = cfg.MODEL.DECODER.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["total_num_feature_levels"] = cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS
        ret["semantic_ce_loss"] = cfg.MODEL.DECODER.TEST.SEMANTIC_ON and cfg.MODEL.DECODER.SEMANTIC_CE_LOSS and ~cfg.MODEL.DECODER.TEST.PANOPTIC_ON
        return ret


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio


    def forward(self, x, mask_features, masks, targets=None):
        """
        :param x: input, a list of multi-scale feature
        :param mask_features: is the per-pixel embeddings with resolution 1/4 of the original image,
        obtained by fusing backbone encoder encoded features. This is used to produce binary masks.
        :param masks: mask in the original image
        :param targets: used for denoising training
        """
        assert len(x) == self.num_feature_levels
        device = x[0].device
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        enc_outputs_class_unselected = self.class_embed(output_memory)
        enc_outputs_coord_unselected = self._bbox_embed(
            output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
        
        topk = self.num_queries
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
        refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                               topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid

        tgt_undetach = torch.gather(output_memory, 1,
                              topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # unsigmoid

        outputs_class, outputs_mask = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), mask_features)
        
        interm_outputs=dict()
        interm_outputs['pred_logits'] = outputs_class
        interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
        interm_outputs['pred_masks'] = outputs_mask

        mask_dict = None

        # Optimization rely on the final layer
        out = {
            'pred_logits': interm_outputs['pred_logits'],
            'pred_masks': interm_outputs['pred_masks'],
            'pred_boxes': interm_outputs['pred_boxes']
        }
        
        out['interm_outputs'] = interm_outputs
        return out, mask_dict, None # losses_moe=None


    def forward_prediction_heads(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        return outputs_class, outputs_mask
        