
"""
ELSE-MoS Training Script based on MaskDINO.
"""
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import time

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    # COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    # SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.analysis import parameter_count, FlopCountAnalysis
from detectron2.data import DatasetCatalog
from detectron2.checkpoint import Checkpointer
from detectron2.engine.hooks import BestCheckpointer
from detectron2.engine.hooks import EvalHook
from detectron2.engine.hooks import PeriodicCheckpointer
import json

from model_zoo.shared.trainer_utils import FinalCheckpointer, GPUMemoryLogger

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.solver import LRMultiplier, WarmupParamScheduler

# modified versions from detectron2
from lunacad.evaluation import COCOEvaluator, SemSegEvaluator

from detectron2.data.datasets import load_coco_json

def load_coco_with_semantic(json_file, image_root, sem_seg_root):
    dataset = load_coco_json(json_file, image_root)
    for d in dataset:
        filename = d["file_name"]
        d["sem_seg_file_name"] = os.path.join(sem_seg_root, os.path.basename(filename))
    return dataset


# SELECT DATASET
DATASET = 'LUL100MT'
# ChangE LU LU-fixed LRO-L4 LUL100MT

_thing_classes = ["crater"] if DATASET not in ['LRO-L4', 'LUL100MT'] else ["lineaments"]
_stuff_classes = ["background", "crater"] if DATASET not in ['LRO-L4', 'LUL100MT'] else ["background", "lineaments"]

DatasetCatalog.register(
    "train2017",
    lambda: load_coco_with_semantic(
        os.path.join("datasets/", DATASET, "annotations/instances_sem_train2017.json"),
        os.path.join("datasets/", DATASET, "train2017"),
        os.path.join("datasets/", DATASET, "annotations/sem/train2017")
    )
)

DatasetCatalog.register(
    "val2017",
    lambda: load_coco_with_semantic(
        os.path.join("datasets/", DATASET, "annotations/instances_sem_val2017.json"),
        os.path.join("datasets/", DATASET, "val2017"),
        os.path.join("datasets/", DATASET, "annotations/sem/val2017")
    )
)

MetadataCatalog.get("train2017").set(
    thing_classes=_thing_classes,
    sem_seg_root=os.path.join("datasets/", DATASET, "annotations/sem/train2017"),
    stuff_classes=_stuff_classes,      # for semantic segmentation
    evaluator_type="coco",
    ignore_label=255   # standard default
)

MetadataCatalog.get("val2017").set(
    thing_classes=_thing_classes,
    sem_seg_root=os.path.join("datasets/", DATASET, "annotations/sem/val2017"),
    stuff_classes=_stuff_classes,      # for semantic segmentation
    evaluator_type="coco",
    ignore_label=255   # standard default
)

# test2017 (merged / held-out splits, LUL100MT only)
for _split in ("test2017"):
    _json = os.path.join("datasets/", DATASET, "annotations/instances_sem_" + _split + ".json")
    if not os.path.exists(_json):
        continue
    if _split in DatasetCatalog.list():
        DatasetCatalog.remove(_split)
        MetadataCatalog.remove(_split)
    DatasetCatalog.register(
        _split,
        lambda split=_split: load_coco_with_semantic(
            os.path.join("datasets/", DATASET, "annotations/instances_sem_" + split + ".json"),
            os.path.join("datasets/", DATASET, split),
            os.path.join("datasets/", DATASET, "annotations/sem/" + split)
        )
    )
    MetadataCatalog.get(_split).set(
        thing_classes=_thing_classes,
        sem_seg_root=os.path.join("datasets/", DATASET, "annotations/sem/" + _split),
        stuff_classes=_stuff_classes,      # for semantic segmentation
        evaluator_type="coco",
        ignore_label=255   # standard default
    )

# LunaCAD
from lunacad import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_decoder_config,
    DetrDatasetMapper,
    NoResizeMapper,
)
import random

from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer,
    HookBase
)
import weakref
import os


def build_lr_scheduler(cfg, optimizer):
    return LRMultiplier(
        optimizer,
        multiplier=WarmupParamScheduler(
            MultiStepParamScheduler(
                values=cfg.SOLVER.LR_MULTIPLIER,
                milestones=cfg.SOLVER.LR_DROP_ITERS,
                num_updates=cfg.SOLVER.MAX_ITER,
            ),
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_length=cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        ),
        max_iter=cfg.SOLVER.MAX_ITER,
    )


# ============================================================================
# Gradient Accumulation Trainers
# ============================================================================

class GradientAccumulationSimpleTrainer(SimpleTrainer):
    """
    SimpleTrainer with gradient accumulation support.

    Args:
        model: model to train
        data_loader: data loader
        optimizer: optimizer
        cfg: configuration node, must contain SOLVER.ACCUMULATION_STEPS
    """
    def __init__(self, model, data_loader, optimizer, cfg):
        super().__init__(model, data_loader, optimizer)
        self.accumulation_steps = cfg.SOLVER.get("ACCUMULATION_STEPS", 1)
        self._current_step = 0  # counts sub-iterations within an accumulation cycle

    def run_step(self):
        """
        Implement gradient accumulation logic:
        1. zero_grad only at the first sub-step of each accumulation cycle
        2. scale loss by 1/accumulation_steps
        3. backward every sub-step
        4. optimizer.step() and write_metrics only at the last sub-step
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # Step 1: zero_grad only at accumulation boundary start
        if self._current_step % self.accumulation_steps == 0:
            self.optimizer.zero_grad()

        # Step 2: forward
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        # Step 3: scale loss for correct gradient magnitude
        losses_scaled = losses / self.accumulation_steps

        # Step 4: anomaly check (inline, same logic as write_metrics)
        if not torch.isfinite(losses_scaled).all():
            raise FloatingPointError(
                f"Loss became infinite or NaN at iteration={self.iter}!\n"
                f"loss_dict = {loss_dict}"
            )

        # Step 5: backward
        losses_scaled.backward()

        self._current_step += 1

        # Step 6: optimizer step and metrics only at accumulation boundary end
        if self._current_step % self.accumulation_steps == 0:
            self.after_backward()
            self.optimizer.step()
            # write_metrics uses the ORIGINAL unscaled loss_dict for logging
            self._write_metrics(loss_dict, data_time)


class GradientAccumulationAMPTrainer(AMPTrainer):
    """
    AMPTrainer with gradient accumulation support.

    Args:
        model: model to train
        data_loader: data loader
        optimizer: optimizer
        cfg: configuration node, must contain SOLVER.ACCUMULATION_STEPS
    """
    def __init__(self, model, data_loader, optimizer, cfg):
        super().__init__(model, data_loader, optimizer)
        self.accumulation_steps = cfg.SOLVER.get("ACCUMULATION_STEPS", 1)
        self._current_step = 0

    def run_step(self):
        """
        Implement gradient accumulation logic with Automatic Mixed Precision.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # Step 1: zero_grad only at accumulation boundary start
        if self._current_step % self.accumulation_steps == 0:
            self.optimizer.zero_grad()

        # Step 2: forward with autocast
        with torch.amp.autocast('cuda', dtype=self.precision):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())
            losses_scaled = losses / self.accumulation_steps

        # Step 3: anomaly check (inline)
        if not torch.isfinite(losses_scaled).all():
            raise FloatingPointError(
                f"Loss became infinite or NaN at iteration={self.iter}!\n"
                f"loss_dict = {loss_dict}"
            )

        # Step 4: backward with scaled loss
        self.grad_scaler.scale(losses_scaled).backward()

        self._current_step += 1

        # Step 5: optimizer step and metrics only at accumulation boundary end
        if self._current_step % self.accumulation_steps == 0:
            self.after_backward()

            # Unscale gradients before clipping (if any) and optimizer step
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # write_metrics uses the ORIGINAL unscaled loss_dict for logging
            self._write_metrics(loss_dict, data_time)


# ============================================================================
# Main Trainer
# ============================================================================

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer with gradient accumulation.
    """
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)

        # -------------------------------------------------------------------
        # Gradient Accumulation: choose appropriate trainer
        # -------------------------------------------------------------------
        accumulation_steps = cfg.SOLVER.get("ACCUMULATION_STEPS", 1)
        if accumulation_steps > 1:
            if cfg.SOLVER.AMP.ENABLED:
                self._trainer = GradientAccumulationAMPTrainer(
                    model, data_loader, optimizer, cfg
                )
            else:
                self._trainer = GradientAccumulationSimpleTrainer(
                    model, data_loader, optimizer, cfg
                )
            if comm.is_main_process():
                logger.info(
                    f"Gradient Accumulation enabled: "
                    f"accumulation_steps={accumulation_steps}, "
                    f"effective batch size = {cfg.SOLVER.IMS_PER_BATCH * accumulation_steps}"
                )
        else:
            self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
                model, data_loader, optimizer
            )
        # -------------------------------------------------------------------

        # self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.scheduler = build_lr_scheduler(cfg, optimizer)

        # =====  Parameters & GFLOPs & Peak Memory =====
        paras_dict = parameter_count(model)
        print("Model Params:", paras_dict[''])
        print("backbone Params:", paras_dict['backbone'])
        print("encoder Params:", paras_dict['sem_seg_head.pixel_decoder'])
        print("decoder Params:", paras_dict['sem_seg_head.predictor'])
        model.eval()
        device = next(model.parameters()).device
        dummy = torch.randn(1, 1, cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            flops = FlopCountAnalysis(model, [{"image": dummy[0], "height": cfg.INPUT.IMAGE_SIZE, "width": cfg.INPUT.IMAGE_SIZE}])
            if cfg.MODEL.DECODER.SEMANTIC_CE_LOSS == False:
                print("GFLOPs:", float(flops.total()) / 1e9)
        except Exception as e:
            print(f"Could not compute GFLOPs: {e}")
        if device.type == "cuda":
            max_mem_mb = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
            print(f"Peak GPU memory (dummy forward): {max_mem_mb:.1f} MB ({max_mem_mb/1024.0:.2f} GB)")
        model.train()                

        # ==================================
        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
        # TODO: release model conversion checkpointer from DINO to ELSE-MoS
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        # TODO: release GPU cluster submit scripts based on submitit for multi-node training

        self.lr_drop_iters = set(cfg.SOLVER.LR_DROP_ITERS)
        self.lr_gamma = cfg.SOLVER.LR_MULTIPLIER


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name,
                                                output_dir=output_folder,
                                                max_dets_per_image=cfg.TEST.MAX_NUM_TARGETS,
                                                area_range_setting=cfg.SOLVER.AREA_RANGE_SETTING,
                                                # 仅 LUL100MT 用检测框对角线长度做尺度划分
                                                diagonal_range_setting=cfg.SOLVER.DIAGONAL_RANGE_SETTING if DATASET == 'LUL100MT' else None))

            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    sem_threshold=cfg.MODEL.SEM_SEG_HEAD.SEM_THRESHOLD
                )
            )
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.DECODER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.DECODER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.DECODER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.DECODER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.DECODER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.DECODER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.DECODER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.DECODER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # coco instance segmentation lsj new baseline
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_detr":
            mapper = DetrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Semantic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed

                lname = module_name.lower()
                pname = module_param_name.lower()
                if "gbc" in lname or "centers" in pname or "log_sigma" in pname or "log_radius" in pname:
                    hyperparams["lr"] = hyperparams["lr"] * 1.

                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        hooks_list = [
            h
            for h in super().build_hooks()
            if not isinstance(h, (EvalHook, PeriodicCheckpointer))
        ]

        eval_period = self.cfg.TEST.EVAL_PERIOD
        evaluators = [self.build_evaluator(self.cfg, name) for name in self.cfg.DATASETS.TEST]
        def _eval():
            self.model.eval()
            with torch.no_grad():
                results = self.test(self.cfg, self.model, evaluators)
            self.model.train()
            return results

        hooks_list.insert(0, AP50BestEvalHook(
            eval_period=eval_period,
            eval_function=_eval,
            checkpointer=self.checkpointer,
            output_dir=self.cfg.OUTPUT_DIR,
            best_metric=self.cfg.TEST.BEST_METRIC,
            main_task=self.cfg.TEST.MAIN_TASK,
            main_metric=self.cfg.TEST.MAIN_METRIC
        ))

        # Save only the final checkpoint at the end of training.
        hooks_list.append(FinalCheckpointer(self.checkpointer))

        # Log peak GPU memory every PRINT_PERIOD iterations.
        print_period = self.cfg.SOLVER.get("PRINT_PERIOD", 20)
        if print_period > 0:
            hooks_list.append(GPUMemoryLogger(period=print_period))

        return hooks_list

    def run_step(self):
        super().run_step()

    def after_step(self):
        super().after_step()      # EvalHook, save best if need
        curr_iter = self.iter

        # update lr and load best model
        if curr_iter in self.lr_drop_iters:
            self.lr_drop_iters.remove(curr_iter)
            if comm.is_main_process():
                self._drop_lr_and_reload_best(curr_iter)
            comm.synchronize()
            with torch.no_grad():
                for p in self.model.parameters():
                    p.data = p.data.to(device='cuda')

    # ----------------------------------
    def _drop_lr_and_reload_best(self, curr_iter):
        gamma = self.lr_gamma
        best_file = os.path.join(self.cfg.OUTPUT_DIR, "model_best.pth")
        checkpoint = torch.load(best_file, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"], strict=False)
        print('==> load best state!')


class AP50BestEvalHook(EvalHook):
    """ EvalHook：save best val2017/bbox/AP50 """
    def __init__(self, eval_period, eval_function, checkpointer, output_dir, best_metric=-1.0, main_task='bbox', main_metric='AP50'):
        super().__init__(eval_period, eval_function)
        self.checkpointer = checkpointer
        self.output_dir = output_dir
        self.best_metric = best_metric
        self.main_task = main_task
        self.main_metric = main_metric
        os.makedirs(output_dir, exist_ok=True)

    def _do_eval(self):
        results = self._func()
        if self.main_task in results.keys():
            metric = results[self.main_task][self.main_metric]
            if metric > self.best_metric:
                self.best_metric = metric
                basename = os.path.basename("model_best")   # "model_best"
                self.checkpointer.save(basename)            # basename

                txt_path = os.path.join(self.output_dir, "best_eval_results.txt")
                with open(txt_path, "w") as f:
                    f.write(f"best {self.main_metric} ({self.main_task}) = {metric:.4f}\n")
                    f.write("-" * 60 + "\n")
                    f.write(json.dumps(results, indent=4))

                print(f"[{self.main_task}-{self.main_metric}-BestEvalHook] New best val2017 AP50={metric:.4f} -> {basename}.pth")
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_decoder_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="MOBA")
    return cfg


def main(args):
    cfg = setup(args)
    # cfg.OUTPUT_DIR="./output-pp"

    print("Command cfg:", cfg)
    if args.eval_only:
        # cfg.OUTPUT_DIR="./output-pp"

        model = Trainer.build_model(cfg)
        # cfg.OUTPUT_DIR="./output-pp"
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()
    # random port
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
