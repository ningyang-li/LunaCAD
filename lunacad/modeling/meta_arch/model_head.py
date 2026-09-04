# ELSE-MoS Head

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn
import torch

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.task3_decoder import build_task3_decoder
from ..pixel_decoder.etca_encoder import build_etca_encoder
from ..pixel_decoder.uagr import wasserstein_diversity_loss, calculate_scale_loss
from ..pixel_decoder.hgc import HGC

@SEM_SEG_HEADS_REGISTRY.register()
class Model_Head(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_predictor: nn.Module,
        use_hgc: bool = False,
        hgc_num_layers: int = 6,
        hgc_dim: int = 256,
        hgc_tau: float = 1.0,
        hgc_weight: float = 0.01,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

        self.num_classes = num_classes

        self.use_hgc = use_hgc
        self.hgc_weight = hgc_weight
        self.hgc = HGC(num_layers=hgc_num_layers, dim=hgc_dim, tau=hgc_tau) if use_hgc else None

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        if cfg.MODEL.MOE.ATTN_TYPE_ENC == "etca":
            pixel_decoder = build_etca_encoder(cfg, input_shape)
        else:
            pixel_decoder = build_dst_encoder(cfg, input_shape)

        transformer_predictor = build_task3_decoder(cfg, transformer_predictor_in_channels, mask_classification=True)
        
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": pixel_decoder,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_predictor": transformer_predictor,
            "use_hgc": getattr(cfg.MODEL, 'HGC', {}).get('ENABLED', False) if hasattr(cfg.MODEL, 'HGC') else False,
            "hgc_num_layers": cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS,
            "hgc_dim": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "hgc_tau": getattr(cfg.MODEL, 'HGC', {}).get('TAU', 1.0) if hasattr(cfg.MODEL, 'HGC') else 1.0,
            "hgc_weight": getattr(cfg.MODEL, 'HGC', {}).get('WEIGHT', 0.01) if hasattr(cfg.MODEL, 'HGC') else 0.01,
        }

    def forward(self, features, mask=None,targets=None):
        return self.layers(features, mask,targets=targets)

    def layers(self, features, mask=None,targets=None):
        mask_features, transformer_encoder_features, multi_scale_features, losses_moe_enc, loss_sampling_rank, uagr_intermediates, all_centers, all_features, all_attentions = self.pixel_decoder.forward_features(features, mask)
        outputs, mask_dict, losses_moe_dec = self.predictor(
            multi_scale_features, mask_features, mask, targets=targets
        )

        # UAGR
        if len(uagr_intermediates) > 0 and hasattr(self.pixel_decoder, 'transformer') and \
           hasattr(self.pixel_decoder.transformer, 'encoder'):
            # Diversity Loss
            centers_list = []
            for layer in self.pixel_decoder.transformer.encoder.layers:
                if hasattr(layer, 'uagr') and layer.uagr is not None:
                    centers_list.append(layer.uagr.centers)
            if len(centers_list) > 0:
                all_centers_cat = torch.cat(centers_list, dim=0)
                outputs['loss_uagr_div'] = wasserstein_diversity_loss(all_centers_cat)

            # Scale Consistency Loss, layer-wise and average
            # based on log_sigma of each layer
            scale_loss = 0
            for idx, inter in enumerate(uagr_intermediates):
                layer = self.pixel_decoder.transformer.encoder.layers[idx]
                if hasattr(layer, 'uagr') and layer.uagr is not None:
                    scale_loss += calculate_scale_loss(
                        inter['att'], inter['dif'], layer.uagr.log_sigma
                    )
            outputs['loss_uagr_scale'] = scale_loss / len(uagr_intermediates)

        # HGC
        if self.use_hgc and len(all_centers) > 1:
            outputs['loss_hgc'] = self.hgc(all_centers, all_features, all_attentions)
            
        losses_moe = losses_moe_enc
        if losses_moe_dec is not None:
            for k in losses_moe.keys():
                losses_moe[k] += losses_moe_dec[k]
        outputs.update(losses_moe)
        outputs.update(loss_sampling_rank)

        return outputs, mask_dict



        
