from __future__ import print_function
import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torchsummary import summary

class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding):
		super(ResBlock, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.padding = padding
		
		self.convblock1 = nn.Sequential(
			nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
			nn.BatchNorm2d(self.out_channels),
			nn.ReLU()
		)
		self.convblock2 = nn.Sequential(
			nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
			nn.BatchNorm2d(self.out_channels),
			nn.ReLU()
		)
	
	def forward(self, x):
		y = self.convblock1(x)
		y = self.convblock2(y)
		
		return y

class CustomResidualNet(nn.Module):
    def __init__(self, dropout_value=0, num_of_inp_channels=3, num_of_op_channels=10):
        super(CustomResidualNet, self).__init__()

        self.dropout_value = dropout_value
        self.num_of_channels = num_of_inp_channels
        self.num_of_op_channels = num_of_op_channels
        self.number_of_kernels = [64, 128, 128, 256, 512, 512]

        # Input Block
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=self.num_of_channels, out_channels=self.number_of_kernels[0], kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(self.number_of_kernels[0]),
			nn.ReLU()
        ) # input_size = 32x32x3, output_size = 32x32x64, RF = 3x3

        # LAYER 1
        self.layer1_x = nn.Sequential(
            nn.Conv2d(in_channels=self.number_of_kernels[0], out_channels=self.number_of_kernels[1], kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
			nn.BatchNorm2d(self.number_of_kernels[1]),
            nn.ReLU()
        ) # input_size = 32x32x64, output_size = 32x32x128, RF = 5x5
		
        # RESIDUAL BLOCK 1
        self.resblock1 = ResBlock(in_channels=self.number_of_kernels[1], out_channels=self.number_of_kernels[2], kernel_size=(3,3), padding=1)
		# input_size = 32x32x128, output_size = 32x32x128, RF = 5x5, 9x9
        
        # LAYER 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=self.number_of_kernels[2], out_channels=self.number_of_kernels[3], kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
			nn.BatchNorm2d(self.number_of_kernels[3]),
            nn.ReLU()
        ) # input_size = 32x32x128, output_size = 16x16x256, RF = 8x8, 12x12
		
        # LAYER 3
        self.layer3_x = nn.Sequential(
            nn.Conv2d(in_channels=self.number_of_kernels[3], out_channels=self.number_of_kernels[4], kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
			nn.BatchNorm2d(self.number_of_kernels[4]),
            nn.ReLU()
        ) # input_size = 16x16x256, output_size = 8x8x512, RF = 
		
        # RESIDUAL BLOCK 1
        self.resblock2 = ResBlock(in_channels=self.number_of_kernels[4], out_channels=self.number_of_kernels[5], kernel_size=(3,3), padding=1)
		# input_size = 8x8x512, output_size = 8x8x512, RF = 
        
        # OUTPUT LAYER
        self.max_pool = nn.MaxPool2d(4, 2) # input_size = 8x8x512, output_size = 1x1x512, RF = 
		
        self.fc_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.number_of_kernels[5], out_channels=self.num_of_op_channels, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x512, output_size = 1x1x10, RF =  
        
        self.rb1 = nn.Sequential()
        self.rb2 = nn.Sequential()


    def forward(self, inp):
        x0 = self.preplayer(inp)
		
        x = self.layer1_x(x0)
        r1 = self.resblock1(x)
        y1 = r1 + x
        y1 = self.rb1(y1)

        y2 = self.layer2(y1)

        x3 = self.layer3_x(y2)
        r2 = self.resblock2(x3)
        y3 = r2 + x3
        y3 = self.rb2(y3)

        y4 = self.max_pool(y3)     
        y5 = self.fc_layer(y4)

        y5 = y5.view(-1, 10)
        #y5 = nn.Softmax(dim=-1)(y5)
        return y5