# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_decoder_config(cfg):
    """
    Add config for DECODER.
    """
    # NOTE: configs from original mask2former
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "DECODER_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.OUTPUT_DIR = "./output"

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # decoder config
    cfg.MODEL.DECODER = CN()
    cfg.MODEL.DECODER.LEARN_TGT = False

    # loss
    cfg.MODEL.DECODER.PANO_BOX_LOSS = False
    cfg.MODEL.DECODER.SEMANTIC_CE_LOSS = False
    cfg.MODEL.DECODER.DEEP_SUPERVISION = True
    cfg.MODEL.DECODER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.DECODER.CLASS_WEIGHT = 4.0
    cfg.MODEL.DECODER.DICE_WEIGHT = 5.0
    cfg.MODEL.DECODER.MASK_WEIGHT = 5.0
    cfg.MODEL.DECODER.BOX_WEIGHT = 5.
    cfg.MODEL.DECODER.GIOU_WEIGHT = 2.

    # cost weight
    cfg.MODEL.DECODER.COST_CLASS_WEIGHT = 4.0
    cfg.MODEL.DECODER.COST_DICE_WEIGHT = 5.0
    cfg.MODEL.DECODER.COST_MASK_WEIGHT = 5.0
    cfg.MODEL.DECODER.COST_BOX_WEIGHT = 5.
    cfg.MODEL.DECODER.COST_GIOU_WEIGHT = 2.

    # transformer config
    cfg.MODEL.DECODER.NHEADS = 8
    cfg.MODEL.DECODER.DROPOUT = 0.1
    cfg.MODEL.DECODER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DECODER.ENC_LAYERS = 0
    cfg.MODEL.DECODER.DEC_LAYERS = 6
    cfg.MODEL.DECODER.INITIAL_PRED = True
    cfg.MODEL.DECODER.PRE_NORM = False
    cfg.MODEL.DECODER.BOX_LOSS = True
    cfg.MODEL.DECODER.HIDDEN_DIM = 256
    cfg.MODEL.DECODER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.DECODER.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.DECODER.TWO_STAGE = True
    cfg.MODEL.DECODER.INITIALIZE_BOX_TYPE = 'no'  # ['no', 'bitmask', 'mask2box']
    cfg.MODEL.DECODER.DN="seg"
    cfg.MODEL.DECODER.DN_NOISE_SCALE=0.4
    cfg.MODEL.DECODER.DN_NUM=100
    cfg.MODEL.DECODER.PRED_CONV=False

    cfg.MODEL.DECODER.EVAL_FLAG = 1

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
    cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 4
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER = 'high2low'  # ['low2high', 'high2low'] high2low: from high level to low level
    cfg.MODEL.SEM_SEG_HEAD.SEM_THRESHOLD = 0.5 # threshold for semantic segmenatation mask
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_THRESHOLD = 0.99 # threshold for semantic segmenatation mask
    
    #####################

    # DECODER inference config
    cfg.MODEL.DECODER.TEST = CN()
    cfg.MODEL.DECODER.TEST.TEST_FOUCUS_ON_BOX = False
    cfg.MODEL.DECODER.TEST.SEMANTIC_ON = True
    cfg.MODEL.DECODER.TEST.INSTANCE_ON = False
    cfg.MODEL.DECODER.TEST.PANOPTIC_ON = False
    cfg.MODEL.DECODER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.DECODER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.DECODER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.DECODER.TEST.PANO_TRANSFORM_EVAL = True
    cfg.MODEL.DECODER.TEST.PANO_TEMPERATURE = 0.06
    cfg.MODEL.DECODER.TEST.MULTIPLE_CLS_EVAL = False
    # cfg.MODEL.DECODER.TEST.EVAL_FLAG = 1

    # NMS post-processing for instance (detection & segmentation) predictions.
    # Disabled by default; enable it for datasets with many duplicate
    # predictions around the same target (e.g. LUL100MT).
    cfg.MODEL.DECODER.TEST.NMS_ENABLED = False
    # IoU basis for NMS: "bbox" (box IoU) or "segm" (mask IoU)
    cfg.MODEL.DECODER.TEST.NMS_TYPE = "bbox"
    # instances with IoU >= NMS_THRESH are suppressed as duplicates
    cfg.MODEL.DECODER.TEST.NMS_THRESH = 0.5

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.DECODER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "DECODER_Encoder"

    # transformer module
    cfg.MODEL.DECODER.TRANSFORMER_DECODER_NAME = "DECODER_Decoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.DECODER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.DECODER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.DECODER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    cfg.Default_loading=True  # a bug in my d2. resume use this; if first time ResNet load, set it false

    # TEST
    cfg.TEST.MAX_NUM_TARGETS = 217 # max number of targets in an image of a dataset
    cfg.TEST.BEST_METRIC = -1.0
    cfg.TEST.MAIN_TASK = "bbox"
    cfg.TEST.MAIN_METRIC = "AP50"

    #####################
    # MoE configs
    cfg.MODEL.MOE = CN()
    cfg.MODEL.MOE.ATTN_TYPE_ENC = "moba"
    cfg.MODEL.MOE.FFN_TYPE_ENC = "ffn"
    cfg.MODEL.MOE.NUM_EXPERTS = 4
    cfg.MODEL.MOE.K = 2
    cfg.MODEL.MOE.NOISY_GATING = True
    cfg.MODEL.MOE.W_TOPK_LOSS = 0.01
    cfg.MODEL.MOE.W_SWITCH_LOSS = 0.01
    cfg.MODEL.MOE.W_Z_LOSS = 0.01
    cfg.MODEL.MOE.ACC_AUX_LOSS = True
    cfg.MODEL.MOE.N_POINTS = 4
    cfg.MODEL.MOE.W_SAMPLING_RANK = 0.0001

    # Print Losses
    cfg.SOLVER.PRINT_PERIOD = 5
    cfg.SOLVER.LR_DROP_ITERS = [10000, 15000, 17500, 19000]   
    cfg.SOLVER.LR_MULTIPLIER = [1.0, 0.5, 0.25, 0.125, 0.1]
    cfg.SOLVER.AREA_RANGE_SETTING = [10, 50]
    # LUL100MT 专用: 不为 None 时, 用检测框对角线长度 sqrt(w^2+h^2) 代替 area
    # 划分 small/medium/large, 阈值单位为像素。仅 LUL100MT 的 yaml 会设置该项。
    cfg.SOLVER.DIAGONAL_RANGE_SETTING = None
    cfg.SOLVER.ACCUMULATION_STEPS = 4

    # ===================== UAGR Config =====================
    cfg.MODEL.UAGR = CN()
    cfg.MODEL.UAGR.ENABLED = False
    cfg.MODEL.UAGR.NUM_BALLS = 32
    cfg.MODEL.UAGR.USE_DIAG_COV = True
    cfg.MODEL.UAGR.TAU = 1.0
    # four schemes: "post_attn", "post_ffn", "pre_attn", "parallel"
    cfg.MODEL.UAGR.POSITION = "post_attn"
    cfg.MODEL.UAGR.W_DIV = 0.01
    cfg.MODEL.UAGR.W_SCALE = 0.01
    # =========================================================

    cfg.MODEL.HGC = CN()
    cfg.MODEL.HGC.ENABLED = True          
    cfg.MODEL.HGC.TAU = 1.0              
    cfg.MODEL.HGC.WEIGHT = 0.01          
        
