#!/usr/bin/env python3
import os, cv2, numpy as np
import argparse
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from lunacad import add_decoder_config
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BitMasks
from pycocotools import mask as mask_util
import torch
import shutil
from detectron2.structures import polygons_to_bitmask


MAX_INST = 600                 
np.random.seed(42)            
COLOR_MAP = np.random.randint(64, 256, (MAX_INST, 3), dtype=np.uint8)
# COLOR_MAP[0] = [0, 0, 0]

DATASET = 'LUL100MT'
# CraterDANet  ChangE LROC-LM  LU5M6TGT  LU5M6TGTm  LU5M6TGT2  LU LU-fixed  LRO-L4  LUL100MT
_thing_classes = ["crater"] if DATASET not in ['LRO-L4', 'LUL100MT'] else ["lineaments"]
_stuff_classes = ["crater"] if DATASET not in ['LRO-L4', 'LUL100MT'] else ["lineaments"]
_ann_json = "val2017" if DATASET not in ['LUL100MT'] else "test2017"

ALPHA = 0.3
SCORE_THRESH = 0.1
TOP_K = {'ChangE': 217,  'CraterDANet': 108,  'LU': 516,  'LU-fixed': 516,  'LRONAC': 400,  'MDCD': 104,  'LRO-L4': 47, 'LUL100MT': 24}[DATASET]
# TOP_K = {'ChangE': 155,  'CraterDANet': 63,  'LU': 289,  'LRONAC': 459,  'MDCD': 52,  'LRO-L4': 13}[DATASET]

# parse optional config file path from command line
parser = argparse.ArgumentParser(description="LunaCAD predictor")
parser.add_argument(
    "--config-file",
    type=str,
    default=None,
    help="Path to the LunaCAD config file (default: configs/lunacad_<DATASET>.yaml)",
)
parser.add_argument(
    "--weights",
    type=str,
    default=None,
    help="Path to model weights (default: output/model_best.pth)",
)
parser.add_argument(
    "--vis-dir",
    type=str,
    default="vis",
    help="Path to save prediction",
)
parser.add_argument(
    "--image-name",
    type=str,
    default=None,
    help="Only predict the image whose basename matches this name",
)
args = parser.parse_args()

config_file = args.config_file if args.config_file else "configs/lunacad_" + DATASET + ".yaml"

# regist dataset
register_coco_instances(
    _ann_json,
    {},
    os.path.join("datasets", DATASET, "annotations/instances_sem_" + _ann_json + ".json"),
    os.path.join("datasets", DATASET, _ann_json)
) # train2017 val2017
MetadataCatalog.get(_ann_json).thing_classes = _thing_classes
MetadataCatalog.get(_ann_json).stuff_classes = _stuff_classes


# load model
cfg = get_cfg()
add_decoder_config(cfg)
cfg.merge_from_file(config_file)
# cfg.merge_from_file("configs/maskdino_R50_task3.yaml")
cfg.MODEL.WEIGHTS = args.weights if args.weights else "output/model_best.pth"
cfg.MODEL.DECODER.TEST.SEMANTIC_ON = True
cfg.MODEL.DECODER.TEST.INSTANCE_ON = True
cfg.MODEL.DECODER.TEST.PANOPTIC_ON = False
predictor = DefaultPredictor(cfg)

# traverse samples
test_dicts = DatasetCatalog.get(_ann_json)
meta = MetadataCatalog.get(_ann_json)
out_dir = args.vis_dir
os.makedirs(out_dir, exist_ok=True)


index = 0
start_index = 0
stop_index = -1

for d in test_dicts:
    if index < start_index:
        index += 1
        continue
        
    im = cv2.imread(d["file_name"])
    file_name = os.path.basename(d['file_name'])
    if args.image_name is not None and file_name != args.image_name:
        index += 1
        continue
    '''
    MOBA-Net
    ["fine-4500_13500_3000.png". "fine-0_12500_5000.png", "147.8795207211168,149.25140156095404,9.05903044959876,10.430911289436077.png", '-152.56238320325375,-151.1905023634164,41.98417060569416,43.35605144553147.png"]

    LunaCAD
    ["fine-4500_13500_3000.png", "-152.56238320325375,-151.1905023634164,41.98417060569416,43.35605144553147.png", "lt_lon-049.98798_lat+30.90363_rb_lon-047.03316_lat+27.94881.png"]
    '''
    # if file_name not in ["fine-4500_13500_3000.png", "-152.56238320325375,-151.1905023634164,41.98417060569416,43.35605144553147.png", "lt_lon-049.98798_lat+30.90363_rb_lon-047.03316_lat+27.94881.png"]:
    #     index += 1
    #     continue

    im_rgb = im[:, :, ::-1]
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    scores = instances.scores           # tensor(N,)
    masks = instances.pred_masks        # Tensor(N,H,W)
    instances.pred_masks = masks

    # ==================== 新增：按置信度阈值过滤 ====================
    if len(instances) > 0:
        scores = instances.scores
        
        # Step 1: Top-K 筛选（保留置信度最高的 K 个）
        if TOP_K is not None and TOP_K > 0 and len(scores) > TOP_K:
            # 获取前K个最高分的索引
            topk_indices = torch.topk(scores, k=TOP_K).indices
            instances = instances[topk_indices]
            scores = instances.scores  # 更新scores
            print(f"{file_name}: Top-K filtering, kept {TOP_K}/{len(scores)+TOP_K} highest scores")

        # Step 2: 置信度阈值过滤
        keep = scores > SCORE_THRESH
        instances = instances[keep]
        
        print(f"{file_name}: {len(scores)} instances after Top-K, {keep.sum().item()} passed thresh={SCORE_THRESH}")
    # ============================================================
    
    cv2.imwrite(os.path.join(out_dir, f"{file_name}"), im_rgb)
    
    if len(instances) > 0:
        canvas = im_rgb.copy().astype(np.float32)
    
        for idx in range(len(instances)):
            mask = instances.pred_masks[idx].numpy().astype(bool)  
            if mask.sum() == 0:
                continue
    
            color = COLOR_MAP[idx % MAX_INST]    
            color = color.astype(np.float32)
            canvas[mask] = canvas[mask] * (1 - ALPHA) + color * ALPHA

            edge = mask & (
                ~np.pad(mask, ((1, 0), (0, 0)), mode='constant')[:-1, :] |
                ~np.pad(mask, ((0, 1), (0, 0)), mode='constant')[1:, :] |
                ~np.pad(mask, ((0, 0), (1, 0)), mode='constant')[:, :-1] |
                ~np.pad(mask, ((0, 0), (0, 1)), mode='constant')[:, 1:]
            )
            canvas[edge] = color

        # instance visualization
        vis_ins = np.clip(canvas, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{file_name}_pred_instance_no_text.png"),
                    cv2.cvtColor(vis_ins, cv2.COLOR_RGB2BGR))
    
    # detection bboxes
    if len(instances) > 0:
        img_bbox = im_rgb.copy()
        boxes = instances.pred_boxes.tensor.numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_bbox, (x1, y1), (x2, y2),
                          color=(0, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(out_dir, f"{file_name}_pred_bbox.png"), img_bbox)
    
    # semantic segmentation
    if "sem_seg" in outputs:
        sem_seg = outputs['sem_seg'].squeeze().cpu().numpy()
        binary = np.where(sem_seg, 255, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{file_name}_pred_semantic.png"), binary)

    # ====================================================================================
    # GT
    anns = d["annotations"]

    lbl_instance = im_rgb.copy().astype(np.float32)
    for lbl_idx, ann in enumerate(anns):
        height, width = im.shape[:2]
        mask_bool = polygons_to_bitmask(ann["segmentation"], height, width)
        if mask_bool.sum() == 0:
            continue
    
        color = COLOR_MAP[lbl_idx % MAX_INST].astype(np.float32)
    
        lbl_instance[mask_bool] = lbl_instance[mask_bool] * (1 - ALPHA) + color * ALPHA
        edge = mask_bool & (
            ~np.pad(mask_bool, ((1, 0), (0, 0)), constant_values=False)[:-1, :] |
            ~np.pad(mask_bool, ((0, 1), (0, 0)), constant_values=False)[1:, :] |
            ~np.pad(mask_bool, ((0, 0), (1, 0)), constant_values=False)[:, :-1] |
            ~np.pad(mask_bool, ((0, 0), (0, 1)), constant_values=False)[:, 1:]
        )
        lbl_instance[edge] = color
    
    cv2.imwrite(os.path.join(out_dir, f"{file_name}_label_instance.png"),
                cv2.cvtColor(lbl_instance.astype(np.uint8), cv2.COLOR_RGB2BGR))

    lbl_bbox = im_rgb.copy()
    for ann in anns:
        x, y, w, h = map(int, ann["bbox"])
        cv2.rectangle(lbl_bbox, (x, y), (x + w, y + h),
                      color=(0, 255, 0), thickness=1)
    cv2.imwrite(os.path.join(out_dir, f"{file_name}_label_bbox.png"),
                cv2.cvtColor(lbl_bbox, cv2.COLOR_RGB2BGR))

    h, w = im.shape[:2]
    sem_label = np.zeros((h, w), dtype=np.uint8)   
    for ann in anns:
        mask_bool = polygons_to_bitmask(ann["segmentation"], h, w)
        sem_label[mask_bool] = 255                 
    
    cv2.imwrite(os.path.join(out_dir, f"{file_name}_label_semantic.png"), sem_label)

    # GT and detection bboxes
    if len(instances) > 0:
        img_bbox_all = im_rgb.copy()
        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            cv2.rectangle(img_bbox_all, (x, y), (x + w, y + h),
                          color=(0, 255, 0), thickness=1)
    
        boxes = instances.pred_boxes.tensor.numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_bbox_all, (x1, y1), (x2, y2),
                          color=(0, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(out_dir, f"{file_name}_gt_pred_bbox.png"), img_bbox_all)

    index += 1
    if index == stop_index:
        break
        
print("Visualization Done and Save in ", out_dir)










