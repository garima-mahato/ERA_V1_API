from __future__ import print_function
import math
import numpy
import torch
from torchvision import datasets
from torchvision import transforms as tf
from .data_augmenter import *

def generate_transforms(means, stdevs, is_albumentation=False, is_augment=True):
  # Train Phase transformations
  if is_albumentation:
    train_transforms = AlbumentationTransformations(means, stdevs)
  elif is_augment:
    train_transforms = augment_data(means, stdevs)
  else:
    train_transforms = tf.Compose([
	    tf.ToTensor(),
	    tf.Normalize(means, stdevs)
	  ])

  # Test Phase transformations
  test_transforms = tf.Compose([
    tf.ToTensor(),
    tf.Normalize(means, stdevs)
  ])

  return train_transforms, test_transforms

def generate_cifar10_train_test_dataset(data_path, train_transforms, test_transforms):
  train = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transforms)
  test = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transforms)

  return train, test

def generate_custom_transforms(means, stdevs):
  # Train Phase transformations
  augmentation = tf.RandomApply([
    tf.RandomHorizontalFlip(),
    tf.RandomRotation(10),
    tf.RandomResizedCrop(64)], p=.8)
  
  train_transforms = tf.Compose([
    tf.Lambda(lambda x: x.convert("RGB")),
    augmentation,
    tf.ToTensor(),
    tf.Normalize(means, stdevs)])
  
  test_transforms = tf.Compose([
    tf.Lambda(lambda x: x.convert("RGB")),
    tf.ToTensor(),
    tf.Normalize(means, stdevs)])

  return train_transforms, test_transforms

def generate_custom_train_test_dataset(train_data_path, test_data_path, train_transforms, test_transforms):
  train = datasets.ImageFolder(train_data_path, transform=train_transforms)
  test = datasets.ImageFolder(test_data_path, transform=test_transforms)

  return train, test

def generate_train_test_loader(data_path, SEED, means, stdevs, is_albumentation=False, batch_size=128, is_augment=True, is_custom=False, test_data_path=None):
  if not is_custom:
    train_transforms, test_transforms = generate_transforms(means, stdevs, is_albumentation=False, is_augment=True)
    train, test = generate_cifar10_train_test_dataset(data_path, train_transforms, test_transforms)
  else:
    train_transforms, test_transforms = generate_custom_transforms(means, stdevs)
    train, test = generate_custom_train_test_dataset(data_path, test_data_path, train_transforms, test_transforms)

  # CUDA?
  cuda = torch.cuda.is_available()

  # For reproducibility
  torch.manual_seed(SEED)

  if cuda:
      torch.cuda.manual_seed(SEED)

  # dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

  # train dataloader
  train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

  return train_loader, test_loader
