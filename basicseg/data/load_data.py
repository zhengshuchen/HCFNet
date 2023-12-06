import torch
import os
import torch.utils.data as Data
import numpy as np
import cv2
from basicseg.data.aug_fn import *
from basicseg.utils.registry import DATASET_REGISTRY
from basicseg.loss.boundary_loss import get_dist_map
"""
//train
        //images
        //masks
//test
        //images
        //masks

"""
class Basedataset(Data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.image_root = os.path.join(opt['data_root'], 'images')
        self.mask_root = os.path.join(opt['data_root'], 'masks')
        self.images = os.listdir(self.image_root)
        self.get_name = opt.get('get_name', False)
        self.bd_loss = opt.get('bd_loss', False)
    def __len__(self):
        return len(self.images)
    def setup_transform_fn(self):
        return None
    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.images[index])
        mask_path = image_path.replace(self.image_root, self.mask_root)
        img = np.array(cv2.imread(image_path, cv2.IMREAD_COLOR))
        mask = np.array(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
        mask = (mask / 255.).astype(np.float32)
        transform_fn = self.setup_transform_fn()
        if transform_fn:
            aug_inputs = transform_fn(image=img, mask=mask)
            img_aug, mask_aug = aug_inputs['image'], aug_inputs['mask']
        if self.bd_loss:
            dist_map = get_dist_map(mask_aug)
        if len(mask_aug.shape) == 2:
            mask_aug.unsqueeze_(dim=0)
        if self.get_name:
            return img_aug, mask_aug, self.images[index]
        elif self.bd_loss:
            return img_aug, mask_aug, dist_map
        else:
            return img_aug, mask_aug

@DATASET_REGISTRY.register()
class Dataset_test(Basedataset):
    def __init__(self, opt):
        super().__init__(opt)
    def setup_transform_fn(self):
        return aug_transform_test(self.opt)

@DATASET_REGISTRY.register()
class Dataset_aug_s(Basedataset):
    def __init__(self, opt):
        super().__init__(opt)
    def setup_transform_fn(self):
        return aug_transform_s(self.opt)

@DATASET_REGISTRY.register()
class Dataset_aug_m(Basedataset):
    def __init__(self, opt):
        super().__init__(opt)
    def setup_transform_fn(self):
        return aug_transform_m(self.opt)

@DATASET_REGISTRY.register()
class Dataset_aug_bac(Basedataset):
    def __init__(self, opt):
        super().__init__(opt)
    def setup_transform_fn(self):
        return aug_transform_bac(self.opt)


@DATASET_REGISTRY.register()
class Dataset_infer(Data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.image_root = opt['data_root']
        self.images = os.listdir(self.image_root)
        self.get_name = opt.get('get_name', False)
    def __len__(self):
        return len(self.images)
    def setup_transform_fn(self):
        return aug_transform_test(opt=self.opt)
    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.images[index])
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        transform_fn = self.setup_transform_fn()
        if transform_fn:
            aug_inputs = transform_fn(image=img)
            img_aug = aug_inputs['image']

        if self.get_name:
            return img_aug, self.images[index]
        else:
            return img_aug
