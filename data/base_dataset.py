import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from typing import List

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def _get_base_transform_list(opt) -> List[torch.nn.Module]:
    transform_list = []
    if opt.resize_or_crop == 'resize_and_random_crop':
        transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'resize_and_center_crop':
        transform_list.append(transforms.Resize(opt.fineSize, Image.BICUBIC))
        transform_list.append(transforms.CenterCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomVerticalFlip())

    return transform_list

def get_transform(opt):
    transform_list = _get_base_transform_list(opt)

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_half_transform(opt):
    transform_list = _get_base_transform_list(opt)

    transform_list.append(transforms.CenterCrop(opt.fineSize // 2))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

# def get_half_transform(opt):
#     transform_list1 = []
#     transform_list2 = []
#     transform_list1.append(transforms.RandomCrop(opt.fineSize))
#     transform_list2.append(transforms.Lambda(
#         lambda img: __scale_width_then_half(img, opt.fineSize)))
#     if opt.isTrain and not opt.no_flip:
#         transform_list1.append(transforms.RandomHorizontalFlip())
#         transform_list2.append(transforms.RandomHorizontalFlip())
#
#     transform_list1 += [transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5),
#                                             (0.5, 0.5, 0.5))]
#     transform_list2 += [transforms.ToTensor(),
#                         transforms.Normalize((0.5, 0.5, 0.5),
#                                              (0.5, 0.5, 0.5))]
#     return transforms.Compose(transform_list2), transforms.Compose(transform_list1)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

def __scale_width_then_half(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    img = img.resize((w, h), Image.BICUBIC)
    # print(img.size)
    top = np.random.randint(0, int(h/2))
    left = np.random.randint(0, int(w/2))

    img = img.crop((left, top, int(left + w/2), int(top + h/2)))
    # print(img.size)
    return img
