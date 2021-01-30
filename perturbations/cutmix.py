"""
Props to https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L228
"""

import numpy as np
import torch
import torch.nn.functional as F


def get_rand_bbox(size, lam):

    # Get cutout size
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # Sample location uniformly at random
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Clip
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(images_1, images_2, labels_1, labels_2, beta=1.0):

    # Determine randomly which is the patch
    if np.random.rand() > 0.5:
        images_1, images_2 = images_2, images_1
        labels_1, labels_2 = labels_2, labels_1

    # Randomly sample lambda from beta distribution
    lam = np.random.beta(beta, beta)

    # Get bounding box
    bbx1, bby1, bbx2, bby2 = get_rand_bbox(images_1.shape, lam)

    # Cut and paste images and labels
    images, labels = images_1.clone(), labels_1.clone()
    images[:, :, bbx1:bbx2, bby1:bby2] = images_2[:, :, bbx1:bbx2, bby1:bby2]
    labels[:, :, bbx1:bbx2, bby1:bby2] = labels_2[:, :, bbx1:bbx2, bby1:bby2]
    return images, labels


@torch.no_grad()
def cutmix_combine(images_1, images_2, labels_1, labels_2, beta=1.0):
    """ Transfers style of style images to content images. Assumes input 
        is a PyTorch tensor with a batch dimension."""
    B, sC, sH, sW = images_1.shape
    B, tC, tH, tW = images_2.shape
    if (sH != tH) or (sW != tW):
        images_1 = F.interpolate(images_1, size=(tH, tW), mode='bicubic')
        labels_1 = F.interpolate(
            labels_1.float(), size=(tH, tW), mode='nearest').long()
    mixed_images, mixed_labels = cutmix(
        images_1, images_2, labels_1, labels_2, beta=1.0)
    return mixed_images, mixed_labels
