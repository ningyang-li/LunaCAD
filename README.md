# LunaCAD

[Ningyang Li](https://ningyang-li.github.io//)¹˒², [Hengyu Zhang](https://github.com/zhytest123)¹˒², Kexin Chang¹˒², Qi Wen*¹˒², Xiaolin Tian³, Atta ur Rahman⁴

¹ Technology and Engineering Center for Space Utilization, Chinese Academy of Sciences  
² University of Chinese Academy of Sciences  
³ State Key Laboratory of Lunar and Planetary Sciences (SKLPlanets), Macau University of Science and Technology  
⁴ Department of Geography and Geomatics, University of Peshawar

LunaCAD is a unified detection and segmentation model for lunar surface features (craters and lineaments), built on an encoder-only mask-classification framework with a ResNet-50 backbone. It jointly performs instance segmentation, bounding-box detection, and semantic segmentation on single-channel lunar DOM imagery in a single forward pass.

## Highlights

- **Multi-task in one model**: instance masks, bounding boxes, and semantic segmentation maps from a single network.
- **Mixture-of-Experts encoder (ETCA + SGFE)**: efficient deformable transformer pixel decoder with expert routing and gating-auxiliary losses.
- **Uncertainty-aware components**: **UAGR** (uncertainty-aware Gaussian refinement) and **HGC** modules for robust mask/box localization on densely packed features.
- **LUL100MT dataset**: A lunar lineament multi-task recognition dataset is released, which covers more than 90% lunar lineaments with detection, instance/semantic segmentation annotations. **Download** dataset from [Google drive](https://drive.google.com/file/d/1pwHunMwodJlbzYLLUbXzTGYwbLCEhsYE/view?usp=drive_link) or [BaiduNet disk](https://pan.baidu.com/s/1GaNiVuBZ6JYIt7KWcEsTkQ?pwd=csuu).
- **Flexible dataset support**: works with several lunar crater/lineament benchmarks — ChangE, LU, and LUL100MT (COCO format with per-image semantic masks).

## Requirements

- Python 3.10+, PyTorch 2.x (developed with torch 2.5.1 + CUDA 12.4), detectron 2 0.6
- NVIDIA GPU (single-GPU training is supported)

## Installation

```bash
# 1. detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 2. extra Python deps
pip install timm opencv-python-headless scipy shapely

# 3. deformable-attention CUDA ops (MaskDINO-style)
cd lunacad/modeling/pixel_decoder/ops
python setup.py build install
cd ../../../..

# 4. MoE kernel
cd lunacad/modeling/pixel_decoder/moe
python3 setup.py install
cd ../../../..
```

> **Blackwell GPUs (RTX 5090 / RTX PRO 6000)**: in `lunacad/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu`, replace `AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` with `AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), ...)` (both occurrences) before building. Keep the original code for other GPUs.

## Dataset Preparation

Organize each dataset in COCO format under `datasets/<DATASET_NAME>/`:

```
datasets/LUL100MT/
├── train2017/                  # grayscale PNG tiles
├── val2017/
├── test2017/
└── annotations/
    ├── instances_sem_train2017.json   # COCO instances
    ├── instances_sem_val2017.json
    ├── instances_sem_test2017.json
    └── sem/
        ├── train2017/                 # per-image semantic masks
        └── val2017/
        └── test2017/
```

Supported `<DATASET_NAME>` values (crater datasets: `ChangE`, `LU`, ; lineament dataset: `LUL100MT`) are selected by editing `DATASET` at the top of `train_net.py` (line ~69) and `predict.py` (line ~22).

## Training

```bash
python train_net.py --num-gpus 1 --config-file configs/lunacad_ChangE.yaml
```

Per-dataset configs are provided in `configs/`:

| Dataset   | Config                       | Input size | Task      |
|-----------|------------------------------|-----------:|-----------|
| ChangE    | `configs/lunacad_ChangE.yaml`| 217        | craters   |
| LU        | `configs/lunacad_LU.yaml`    | 416        | craters   |
| LUL100MT  | `configs/lunacad_LUL100MT.yaml` | 896    | lineaments|

Useful options:

```bash
# fine-tune from a checkpoint
python train_net.py --num-gpus 1 --config-file configs/lunacad_ChangE.yaml \
    MODEL.WEIGHTS output/model_best.pth

# override output directory
python train_net.py --num-gpus 1 --config-file configs/lunacad_ChangE.yaml \
    OUTPUT_DIR output-ChangE
```

The trainer saves `model_best.pth` (best `segm/AP50` on the validation set by default, configurable via `TEST.MAIN_TASK` / `TEST.MAIN_METRIC`) and tensorboard logs to `OUTPUT_DIR`. AMP and gradient accumulation (`SOLVER.ACCUMULATION_STEPS`) are supported.

## Evaluation

```bash
python train_net.py --eval-only --num-gpus 1 \
    --config-file configs/lunacad_ChangE.yaml \
    MODEL.WEIGHTS output/model_best.pth \
    OUTPUT_DIR output_vis
```

This reports COCO-style bbox/segm AP (including small/medium/large splits, with area- or diagonal-range scale settings per dataset) plus semantic-segmentation metrics.

## Inference

`predict.py` runs the model on a registered split and saves original image, predicted instance masks, bounding boxes, semantic maps, and GT comparisons to `--vis-dir`:

```bash
python predict.py --weights output/model_best.pth --vis-dir vis
# optional: restrict to one image
python predict.py --weights output/model_best.pth --image-name <file_name.png>
```

## Project Structure

```
├── lunacad/                    # model package
│   ├── lunacad.py              # META_ARCH "LunaCAD"
│   ├── modeling/
│   │   ├── pixel_decoder/      # ETCA encoder, UAGR, HGC, MoE ops
│   │   ├── transformer_decoder/# task3 decoder
│   │   └── meta_arch/          # model head
│   ├── data/                   # dataset mappers
│   └── evaluation/             # COCO / sem-seg evaluators
├── configs/                    # per-dataset configs
├── datasets/                   # COCO-format datasets
├── train_net.py                # training / eval entry point
├── predict.py                  # inference & visualization
```

