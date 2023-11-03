import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random

# using for testset, only resize the img

def aug_transform_test(opt):
    transform_fn = []
    transform_fn.append(A.Resize(opt['img_sz'], opt['img_sz']))
    transform_fn.append(A.Normalize())
    transform_fn.append(ToTensorV2())
    transform_fn = A.Compose(transform_fn)
    return transform_fn

def aug_transform_s(opt):
    transform_fn = []
    transform_fn.append(A.Resize(opt['img_sz'], opt['img_sz']))
    H_Flip = opt.get('H_Flip', False)
    if H_Flip:
        transform_fn.append(A.HorizontalFlip(p=H_Flip))
    V_Flip = opt.get('V_Flip', False)
    if V_Flip:
        transform_fn.append(A.VerticalFlip(p=V_Flip))
    transform_fn.append(A.Normalize())
    transform_fn.append(ToTensorV2())
    transform_fn = A.Compose(transform_fn)
    return transform_fn

def aug_transform_bac(opt):
    img_sz = opt['img_sz']
    random_sz = random.randint(int(img_sz * 0.5), int(img_sz * 1.5))
    transform_fn = []
    transform_fn.append(A.HorizontalFlip(p=0.5))
    transform_fn.append((A.VerticalFlip(p=0.2)))
    transform_fn.append(A.LongestMaxSize(random_sz))
    transform_fn.append(A.PadIfNeeded(img_sz, img_sz))
    transform_fn.append(A.RandomCrop(img_sz, img_sz))
    transform_fn.append(A.Normalize())
    transform_fn.append(ToTensorV2())
    transform_fn = A.Compose(transform_fn)
    return transform_fn

def aug_transform_m(opt):
    img_sz = opt['img_sz']
    transform = [
    A.Resize(img_sz, img_sz),
    A.PadIfNeeded(min_height=img_sz, min_width=img_sz),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.RandomCrop(height=img_sz, width=img_sz),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(),
    ToTensorV2(),
    ]
    transform_fn = A.Compose(transform)
    return transform_fn

def aug_transform_l():
    pass
