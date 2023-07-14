import os
import sys
import time
import kornia

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm.notebook import tqdm
from GPUtil import showUtilization as gpu_usage

from .evaluate import *
from ...models import *
from torch.utils.tensorboard import SummaryWriter
import datetime
# from tensorflow import summary

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model_name, model, criterion_depth, criterion_mask, optimizer, scheduler, train_loader, test_loader, epochs=20, batch_size=32, lr=0.01, save_cp=True, dir_checkpoint='./unet_model/checkpoints/', is_accumulate=False, is_val=True, init_epoch=0, img_step=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_step = 0
    train_losses = []
    val_scores = []
    data_len = len(train_loader)
    accumulation_steps = batch_size // 16

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-{}'.format(model_name, lr, epochs, batch_size))#, flush_secs=30)
    print('{}-lr{}-e{}-{}'.format(model_name, lr, epochs, batch_size))

    # gpu_usage()

    # Start training...
    for epoch in tqdm(range(init_epoch, init_epoch+epochs)):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)
        running_loss = 0.0

        # Switch to train mode
        model.train()

        end = time.time()
		
        optimizer.zero_grad()
        for i, sample_batched in enumerate(tqdm(train_loader)):
            # optimizer.zero_grad()

            # Prepare sample and target
            sample_batched['bg'] = sample_batched['bg'].to(device)
            sample_batched['image'] = sample_batched['image'].to(device)
            sample_batched['depth'] = sample_batched['depth'].to(device)
            sample_batched['mask'] = sample_batched['mask'].to(device)

            # Predict
            if epoch-init_epoch < 1 and i < 1:
              # print(f"GPU Usage after loading sample batches in epoch {epoch} and iteration {i}: ")
              # gpu_usage()
              st = time.time()
            output_depth, output_mask = model(sample_batched)
            if epoch-init_epoch < 1 and i < 1:
               print(f"Actual Prediction time: {time.time() - st} sec")
               del st

            loss_depth = (0.1 * criterion_depth(output_depth, sample_batched['depth'])) + (1.0 * kornia.losses.SSIM(3, reduction='mean')(output_depth, sample_batched['depth'])) + (1.0 * (torch.mean(torch.mean(torch.abs(kornia.filters.SpatialGradient()(output_depth)[:,:,0,:,:].sub(kornia.filters.SpatialGradient()(sample_batched['depth'])[:,:,0,:,:])) + torch.abs(kornia.filters.SpatialGradient()(output_depth)[:,:,1,:,:].sub(kornia.filters.SpatialGradient()(sample_batched['depth'])[:,:,1,:,:])), axis=1))))
            loss_mask = 1.0 * criterion_mask(output_mask, sample_batched['mask']) + (1.0 * kornia.losses.SSIM(3, reduction='mean')(output_mask, sample_batched['mask'])) 
            loss = loss_depth + loss_mask
            
            if is_accumulate:
              loss = loss / accumulation_steps

            # Update step
            losses.update(loss.data.item(), sample_batched['image'].size(0))
            if epoch-init_epoch < 1 and i < 1:
              # print(f"GPU Usage after forward pass in epoch {epoch} and iteration {i}: ")
              # gpu_usage()
              st = time.time()
            loss.backward()
            # optimizer.step()
            if is_accumulate and (i + 1 ) % accumulation_steps == 0:
              optimizer.step()
              optimizer.zero_grad()
            if not is_accumulate:
              optimizer.step()
              optimizer.zero_grad()
            if epoch-init_epoch < 1 and i < 1:
              print(f"Actual Backprop time: {time.time() - st} sec")
              # print(f"GPU Usage after backward pass in epoch {epoch} and iteration {i}: ")
              # gpu_usage()
              del st
            running_loss += loss.item()
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))

            global_step += 1
        
            # Log progress
            if epoch-init_epoch < 1 and i<1:
              st = time.time()
            niter = epoch*N+i
            if i % (N/2) == 0: #(i % (data_len // batch_size)) % 2 == 0:
              # Print to console
              print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
              'ETA {eta}\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'
              .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))
            if i % 1000 == 0:
              writer.add_images('images', sample_batched['image'], global_step)
              writer.add_images('depth/true', sample_batched['depth'], global_step)
              writer.add_images('depth/pred', output_depth, global_step)
              writer.add_images('mask/true', sample_batched['mask'], global_step)
              writer.add_images('mask/pred', output_mask > 0.5, global_step)
            # Log to tensorboard
            writer.add_scalar('Train/Loss', losses.val, global_step)#, niter)
            if epoch-init_epoch < 1 and i < 1:
              # print(f"GPU Usage after 1 sample training in epoch {epoch} and iteration {i}: ")
              # gpu_usage()
              print(f"Train Logging time: {time.time() - st} sec")
              del st

            if i % (N/2) == 0:
              torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}_batch{i}.pth')
        epoch_loss = (running_loss) / N
        print(f"Epoch Loss: {epoch_loss}")
        writer.add_scalar('Loss/Train', epoch_loss, global_step)
        # print(f"GPU Usage after epoch {epoch} : ")
        # gpu_usage()
        train_losses.append(epoch_loss)
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
            except OSError:
                pass
            torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
			
        if is_val: # and global_step % (data_len) == 0:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            val_score, mask_pix_acc, mask_dice, depth_metrics = eval_net(model, test_loader, device)
            val_scores.append(val_score)
            if scheduler is not None:
              scheduler.step(val_score)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            writer.add_scalar('Loss/test', val_score, global_step)
            writer.add_scalar('Mask/test/Mean_IOU', mask_pix_acc, global_step)
            writer.add_scalar('Mask/test/Dice', mask_dice, global_step)
            writer.add_scalar('Depth/test/RMSE', depth_metrics, global_step)

            # writer.add_images('images', image, global_step)
        if epoch % img_step == 0 or epoch == epochs-1:
            writer.add_images('images', sample_batched['image'], global_step)
            writer.add_images('depth/true', sample_batched['depth'], global_step)
            writer.add_images('depth/pred', output_depth, global_step)
            writer.add_images('mask/true', sample_batched['mask'], global_step)
            writer.add_images('mask/pred', output_mask > 0.5, global_step)
        print(f"Total val logging and validation time: {time.time()-end} sec")
              #print(f"Total forward and backward pass time: {(time.time()-end) - batch_time.val} sec")
        if not is_val and scheduler is not None:
          scheduler.step(loss)
        epoch_loss = (running_loss) / N
        print(f"Epoch Loss: {epoch_loss}")
        writer.add_scalar('Loss/Train', epoch_loss, global_step)
        # print(f"GPU Usage after epoch {epoch} : ")
        # gpu_usage()
        # train_losses.append(epoch_loss)
        # if save_cp:
            # try:
                # os.mkdir(dir_checkpoint)
            # except OSError:
                # pass
            # torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
		
    writer.close()
	
    return train_losses, val_scores
