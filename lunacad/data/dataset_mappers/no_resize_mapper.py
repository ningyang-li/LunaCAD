import copy, numpy as np, torch, pycocotools.mask as mask_util
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper as _D2Mapper
from detectron2.structures import BitMasks, Instances
from detectron2.data.transforms import RandomFlip, ResizeShortestEdge, FixedSizeCrop
import detectron2.data.transforms as T
from detectron2.data import detection_utils
import torch
import numpy as np
import copy

class NoResizeMapper:
    def __init__(self, cfg, is_train=True):
        self.img_format = cfg.INPUT.FORMAT  # 通常是 "RGB"

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = detection_utils.read_image(dataset_dict["file_name"], format=self.img_format)
        H, W = image.shape[:2]
        # 1. 原图直接转 tensor（0-255 → 0-255，不做归一化/缩放）
        dataset_dict["image"] = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float()

        # 2. polygon → BitMasks（MaskDINO 必须）
        if "annotations" in dataset_dict:
            masks, classes = [], []
            for anno in dataset_dict["annotations"]:
                seg = anno["segmentation"]
                if isinstance(seg, list):
                    rles = mask_util.frPyObjects(seg, H, W)
                    rle  = mask_util.merge(rles)
                elif isinstance(seg, dict):
                    rle = seg
                else:
                    raise ValueError("unknown seg type")
                m = mask_util.decode(rle)
                masks.append(torch.from_numpy(m))
                classes.append(anno["category_id"])
            instances = Instances((H, W))
            instances.gt_masks   = BitMasks(torch.stack(masks))
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances.gt_boxes   = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = instances
            dataset_dict.pop("annotations", None)
        return dataset_dict