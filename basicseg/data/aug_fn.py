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
    transform_fn = A.Compose(transform_fn, is_check_shapes=False)
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
    transform_fn = A.Compose(transform_fn, is_check_shapes=False)
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
    transform_fn = A.Compose(transform_fn, is_check_shapes=False)
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

def aug_transform_train(opt):
    img_sz = opt['img_sz']  # 确保最终输出大小为 opt['img_sz']
    transform_fn = []

    # 几何变换
    transform_fn.append(A.HorizontalFlip(p=0.5))  # 随机水平翻转
    transform_fn.append(A.VerticalFlip(p=0.1))    # 随机垂直翻转
    transform_fn.append(A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, p=0.5))  # 平移、缩放、旋转

    # 尺寸调整和裁剪（确保输出尺寸固定）
    transform_fn.append(A.Resize(height=img_sz, width=img_sz))  # 强制调整为固定大小
    transform_fn.append(A.PadIfNeeded(min_height=img_sz, min_width=img_sz, border_mode=0, value=0))  # 填充区域的像素值设为0  # 补齐尺寸到固定大小（如果必要）

    # 颜色增强
    transform_fn.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))  # 随机亮度和对比度调整

    # 噪声增强
    transform_fn.append(A.GaussNoise(var_limit=(5.0, 15.0), p=0.3))  # 高斯噪声
    transform_fn.append(A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3))  # 模拟红外传感器噪声

    # 标准化和张量转换
    transform_fn.append(A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))  # 针对 RGB 红外图像的归一化
    transform_fn.append(ToTensorV2())

    transform_fn = A.Compose(transform_fn, is_check_shapes=False)
    return transform_fn

def aug_transform_test_new(opt):
    img_sz = opt['img_sz']  # 确保最终输出大小为 opt['img_sz']
    transform_fn = []

    # 尺寸调整
    transform_fn.append(A.Resize(height=img_sz, width=img_sz))  # 强制调整为固定大小
    transform_fn.append(A.PadIfNeeded(min_height=img_sz, min_width=img_sz, border_mode=0, value=0))  # 填充区域的像素值设为0  # 补齐尺寸到固定大小（如果必要）

    # 标准化和张量转换
    transform_fn.append(A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))  # 针对 RGB 红外图像的归一化
    transform_fn.append(ToTensorV2())

    transform_fn = A.Compose(transform_fn, is_check_shapes=False)
    return transform_fn