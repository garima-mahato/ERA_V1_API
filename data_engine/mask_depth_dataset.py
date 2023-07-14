import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
import os

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        bg, image, depth, mask = sample['bg'], sample['image'], sample['depth'], sample['mask']

        if not _is_pil_image(bg):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(bg)))
        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))
        if not _is_pil_image(mask):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(mask)))

        if random.random() < 0.5:
            bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'bg': bg, 'image': image, 'depth': depth, 'mask': mask}

class maskDepthDatasetMemory(Dataset):
    def __init__(self, data, label, bg_data, resize=None, default_resize=None, transform=None):
        self.data, self.label, self.bg_data = data, label, bg_data
        self.transform = transform
        self.resize = resize
        self.default_resize = default_resize

    def __getitem__(self, idx):
        sample = self.label[idx]
        bg = Image.open( self.bg_data[sample[0]] )
        image = Image.open( BytesIO(self.data[sample[1]]) )
        depth = Image.open( BytesIO(self.data[sample[3]]) ).convert('L')
        mask = Image.open( BytesIO(self.data[sample[2]]) )#.convert(mode='RGB')
        
        if self.resize is not None:
          image = image.resize(self.resize)
          bg = bg.resize(self.resize)
          #mask = mask.resize(self.resize)
        if self.default_resize is not None:
          depth = depth.resize(self.default_resize) #((120, 120))
          mask = mask.resize(self.default_resize)
        
        sample = {'bg': bg, 'image': image, 'depth': depth, 'mask': mask}
        if self.transform: sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.label)

class ToTensor(object):
    def __init__(self,is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        bg, image, depth, mask = sample['bg'], sample['image'], sample['depth'], sample['mask']
        
        bg = self.to_tensor(bg)
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        mask = self.to_tensor(mask)

        return {'bg': bg, 'image': image, 'depth': depth, 'mask': mask}

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])

def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])

