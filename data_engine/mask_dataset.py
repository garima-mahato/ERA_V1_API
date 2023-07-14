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
        image, mask = sample['image'], sample['mask']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(mask):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(mask)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'mask': mask}

class maskDatasetMemory(Dataset):
    def __init__(self, data, label, resize=None, default_resize=None, transform=None):
        self.data, self.label = data, label
        self.transform = transform
        self.resize = resize
        self.default_resize = default_resize

    def __getitem__(self, idx):
        sample = self.label[idx]
        image = Image.open( BytesIO(self.data[sample[2]]) )
        mask = Image.open( BytesIO(self.data[sample[3]]) )
        
        if self.resize is not None:
          image = image.resize(self.resize)
          #mask = mask.resize(self.resize)
        if self.default_resize is not None:
          mask = mask.resize(self.default_resize) #((120, 120))
        sample = {'image': image, 'mask': mask}
        if self.transform: sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.label)

class ToTensor(object):
    def __init__(self,is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask[mask>0] = 1 #* (int(mask.split('/')[-1].split('_')[1][2:]))

        # depth = depth.resize((120, 120))

        # if self.is_test:
        #     depth = self.to_tensor(depth).float() / 1000
        # else:            
        #     depth = self.to_tensor(depth).float() * 1000
        
        # put in expected range
        #depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'mask': mask}

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

