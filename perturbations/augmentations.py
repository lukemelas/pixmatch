import numpy as np
import torch
import cv2
import albumentations as al
from datasets.cityscapes_Dataset import IMG_MEAN


def get_augmentation():
    return al.Compose([
        al.RandomResizedCrop(512, 512, scale=(0.2, 1.)),
        al.Compose([
            # NOTE: RandomBrightnessContrast replaces ColorJitter
            al.RandomBrightnessContrast(p=1),
            al.HueSaturationValue(p=1),
        ], p=0.8),
        al.ToGray(p=0.2),
        al.GaussianBlur(5, p=0.5),
    ])


def augment(images, labels, aug):
    """Augments both image and label. Assumes input is a PyTorch tensor with 
       a batch dimension and values normalized to N(0,1)."""

    # Transform label shape: B, C, W, H ==> B, W, H, C
    labels_are_3d = (len(labels.shape) == 4)
    if labels_are_3d:
        labels = labels.permute(0, 2, 3, 1)

    # Transform each image independently. This is slow, but whatever.
    aug_images, aug_labels = [], []
    for image, label in zip(images, labels):

        # Step 1: Undo normalization transformation, convert to numpy
        image = cv2.cvtColor(image.numpy().transpose(
            1, 2, 0) + IMG_MEAN, cv2.COLOR_BGR2RGB).astype(np.uint8)
        label = label.numpy()  # convert to np

        # Step 2: Perform transformations on numpy images
        data = aug(image=image, mask=label)
        image, label = data['image'], data['mask']

        # Step 3: Convert back to PyTorch tensors
        image = torch.from_numpy((cv2.cvtColor(image.astype(
            np.float32), cv2.COLOR_RGB2BGR) - IMG_MEAN).transpose(2, 0, 1))
        label = torch.from_numpy(label)
        if not labels_are_3d:
            label = label.long()

        # Add to list
        aug_images.append(image)
        aug_labels.append(label)

    # Stack
    images = torch.stack(aug_images, dim=0)
    labels = torch.stack(aug_labels, dim=0)

    # Transform label shape back: B, C, W, H ==> B, W, H, C
    if labels_are_3d:
        labels = labels.permute(0, 3, 1, 2)
    return images, labels
