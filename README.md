<div align="center">    
 
## PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training

[![Paper](http://img.shields.io/badge/paper-arxiv.2105.08128-B31B1B.svg)](https://arxiv.org/abs/2105.08128)
[![Conference](http://img.shields.io/badge/CVPR-2021-4b44ce.svg)](https://arxiv.org/abs/2105.08128)

</div>

<!-- TODO: Add video -->

### Description   
Unsupervised domain adaptation is a promising technique for semantic segmentation and other computer vision tasks for which large-scale data annotation is costly and time-consuming. In semantic segmentation particularly, it is attractive to train models on annotated images from a simulated (source) domain and deploy them on real (target) domains. In this work, we present a novel framework for unsupervised domain adaptation based on the notion of target-domain consistency training. Intuitively, our work is based on the insight that in order to perform well on the target domain, a model’s output should be consistent with respect to small perturbations of inputs in the target domain. Specifically, we introduce a new loss term to enforce pixelwise consistency between the model's predictions on a target image and perturbed version of the same image. In comparison to popular adversarial adaptation methods, our approach is simpler, easier to implement, and more memory-efficient during training. Experiments and extensive ablation studies demonstrate that our simple approach achieves remarkably strong results on two challenging synthetic-to-real benchmarks, GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes. 

### How to run   

#### Dependencies
 - PyTorch (tested on version 1.7.1, but should work on any version)
 - Hydra 1.1: `pip install hydra-core --pre`
 - Other: `pip install albumentations tqdm tensorboard`
 - WandB (optional): `pip install wandb`

#### General
We use Hydra for configuration and Weights and Biases for logging. With Hydra, you can specify a config file (found in `configs/`) with `--config-name=myconfig.yaml`. You can also override the config from the command line by specifying the overriding arguments (without `--`). For example, you can disable Weights and Biases with `wandb=False` and you can name the run with `name=myname`. 

We have prepared example configs for GTA5 and SYNTHIA in `configs/gta5.yaml` and `configs/synthia.yaml`.

#### Data Preparation
To run on GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes, you need to download the respective datasets. Once they are downloaded, you can either modify the config files directly, or organize/symlink the data in the `datasets/` directory as follows: 
```
datasets
├── cityscapes
│   ├── gtFine
│   │   ├── train
│   │   │   ├── aachen
│   │   │   └── ...
│   │   └── val
│   └── leftImg8bit
│       ├── train
│       └── val
├── GTA5
│   ├── images
│   ├── labels
│   └── list
├── SYNTHIA
│   └── RAND_CITYSCAPES
│       ├── Depth
│       │   └── Depth
│       ├── GT
│       │   ├── COLOR
│       │   └── LABELS
│       ├── RGB
│       └── synthia_mapped_to_cityscapes
├── city_list
├── gta5_list
└── synthia_list
```

#### Initial Models
 * For GTA5-to-Cityscapes, we start with a model pretrained on the source (GTA5): [Download](https://github.com/lukemelas/pixmatch/releases/download/v1.0.0/GTA5_source.pth)
 * For SYNTHIA-to-Cityscapes, we start with a model pretrained on ImageNet: [Download](http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth)

#### SYNTHIA-to-Cityscapes
To run a baseline PixMatch model with standard data augmentations, we can use a command such as:
```bash
python main.py --config-name=synthia lam_aug=0.10 name=synthia_baseline
```
It is also easy to run a model with multiple augmentations:
```bash
python main.py --config-name=synthia lam_aug=0.00 lam_fourier=0.10 lam_cutmix=0.10 name=synthia_fourier_and_cutmix
```

#### GTA5-to-Cityscapes

```bash
python main.py --config-name=synthia lam_aug=0.10 name=gta5_baseline
```

#### Evaluation
To evaluate, simply set the `train` argument to False:
```bash
python main.py train=False
```

#### Pretrained models
 * GTA5-to-Cityscapes: [Download](https://github.com/lukemelas/pixmatch/releases/download/v1.0.0/GTA5-to-Cityscapes-checkpoint.pth)
 * SYNTHIA-to-Cityscapes: [Download](https://github.com/lukemelas/pixmatch/releases/download/v1.0.0/SYNTHIA-to-Cityscapes-checkpoint.pth)

To evaluate a pretrained/trained model, you can run: 
```bash
# GTA (default)
CUDA_VISIBLE_DEVICES=3 python main.py train=False wandb=False model.checkpoint=$(pwd)/pretrained/GTA5-to-Cityscapes-checkpoint.pth

# SYNTHIA
CUDA_VISIBLE_DEVICES=3 python main.py --config-name synthia train=False wandb=False model.checkpoint=$(pwd)/pretrained/GTA5-to-Cityscapes-checkpoint.pth
```

#### Citation   
```bibtex
@inproceedings{melaskyriazi2021pixmatch,
  author    = {Melas-Kyriazi, Luke and Manrai, Arjun},
  title     = {PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training},
  booktitle = cvpr,
  year      = {2021}
}
```
