# -*- coding: utf-8 -*-
from PIL import Image, ImageFile
import os
import torch
import torch.utils.data as data

from datasets.cityscapes_Dataset import City_Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def to_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class GTA5_Dataset(City_Dataset):
    def __init__(
        self,
        data_root_path='./datasets/GTA5',
        list_path='./datasets/GTA5/list',
        split='train',
        base_size=769,
        crop_size=769,
        training=True
    ):

        # Args
        self.data_path = data_root_path
        self.list_path = list_path
        self.split = split
        self.base_size = to_tuple(base_size)
        self.crop_size = to_tuple(crop_size)
        self.training = training
        self.class_16 = False
        self.class_13 = False

        # Augmentation
        self.random_mirror = args.random_mirror
        self.random_crop = args.random_crop
        self.resize = args.resize
        self.gaussian_blur = args.gaussian_blur

        # Files
        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        if not os.path.exists(item_list_filepath):
            raise Warning("split must be train/val/trainval/test/all")
        self.image_filepath = os.path.join(self.data_path, "images")
        self.gt_filepath = os.path.join(self.data_path, "labels")
        self.items = [id.strip() for id in open(item_list_filepath)]

        # Label map
        self.id_to_trainid = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
            22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
        }

        # Print
        print("{} num images in GTA5 {} set have been loaded.".format(
            len(self.items), self.split))

    def __getitem__(self, item):
        id = int(self.items[item])
        name = f"{id:0>5d}.png"

        # Open image and label
        image_path = os.path.join(self.image_filepath, name)
        gt_image_path = os.path.join(self.gt_filepath, name)
        image = Image.open(image_path).convert("RGB")
        gt_image = Image.open(gt_image_path)

        # Augmentation
        if (self.split == "train" or self.split == "trainval" or self.split == "all") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)
        return image, gt_image, item