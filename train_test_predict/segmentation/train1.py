import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm.notebook import tqdm

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

def train(model_name, model, criterion, optimizer, scheduler, train_loader, test_loader, epochs=20, batch_size=32, lr=0.01, save_cp=True, dir_checkpoint='./unet_model/checkpoints/', is_accumulate=True, is_val=True, init_epoch=0, img_step=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_list = {'CustomUNet': CustomUNet}
    # model = model_list[model_name](**model_params)
    global_step = 0
    train_losses = []
    val_scores = []
    data_len = len(train_loader)
    accumulation_steps = batch_size // 16

    # Logging
    # train_summary_writer = summary.create_file_writer(train_log_dir)
    # test_summary_writer = summary.create_file_writer(test_log_dir)
    writer = SummaryWriter(comment='{}-lr{}-e{}-{}'.format(model_name, lr, epochs, batch_size))#, flush_secs=30)
    print('{}-lr{}-e{}-{}'.format(model_name, lr, epochs, batch_size))

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
        for i, sample_batched in enumerate(train_loader):
            # optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            true_masks = torch.autograd.Variable(sample_batched['mask'].cuda(non_blocking=True))

            mask_type = torch.float32 if model.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            # Predict
            if epoch < 5 and i==0:
              st = time.time()
            output = model(image)
            if epoch < 5 and i==0:
               print(f"Actual Prediction time: {time.time() - st} sec")
               del st

            loss = criterion(output, true_masks)
            loss = loss / accumulation_steps

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            # optimizer.step()
            if is_accumulate and (i + 1 ) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))

            global_step += 1
        
            # Log progress
            #t1 = time.time()
            niter = epoch*N+i
            if i % (N/2) == 0: #(i % (data_len // batch_size)) % 2 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

            # Log to tensorboard
            writer.add_scalar('Train/Loss', losses.val, global_step)#, niter)
            # with train_summary_writer.as_default():
            #     tf.summary.scalar('loss', loss.item(), step=global_step)

        if is_val: # and global_step % (data_len) == 0:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            val_score = eval_net(model, test_loader, device)
            val_scores.append(val_score)
            if scheduler is not None:
              scheduler.step(val_score)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if model.n_classes > 1:
                writer.add_scalar('Loss/test', val_score, global_step)
            else:
                writer.add_scalar('Dice/test', val_score, global_step)

            # writer.add_images('images', image, global_step)
            if model.n_classes == 1 and epoch % img_step == 0:
                writer.add_images('images', image, global_step)
                writer.add_images('masks/true', true_masks, global_step)
                writer.add_images('masks/pred', torch.sigmoid(output) > 0.5, global_step)
            if epoch == init_epoch:
              print(f"Total logging and validation time: {time.time()-end} sec")
              #print(f"Total forward and backward pass time: {(time.time()-end) - batch_time.val} sec")
        if not is_val and scheduler is not None:
          scheduler.step(loss)
        epoch_loss = (running_loss * accumulation_steps) / N
        print(f"Epoch Loss: {epoch_loss}")
        train_losses.append(epoch_loss)
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
            except OSError:
                pass
            torch.save(model.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
		
    writer.close()
	
    return train_losses, val_score
