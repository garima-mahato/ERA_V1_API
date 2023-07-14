import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
import kornia
import math
import torch
import numpy as np 

from ...loss import *
from .saliency_map import BackPropagation
#from utils import nanmean

EPS = 1e-10

# PyTroch version

SMOOTH = 1e-6

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    eps: float = 1e-6

    dims = (1, 2, 3)

    outputs = torch.sigmoid(outputs)
    intersection = torch.sum((outputs * labels), dims)
    union = torch.sum((outputs + labels), dims)

    iou = intersection / (union + eps)

    miou = torch.mean(iou)

    return miou

# def eval_net(net, loader, device):
    # """Evaluation without the densecrf with the dice coefficient"""
    # net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    # n_val = len(loader)  # the number of batch
    # tot = 0

    # with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        # for batch in loader:
            # imgs, true_masks = batch['image'], batch['mask']
            # imgs = imgs.to(device=device, dtype=torch.float32)
            # true_masks = true_masks.to(device=device, dtype=mask_type)

            # with torch.no_grad():
                # mask_pred = net(imgs)

            # if net.n_classes > 1:
                # tot += F.cross_entropy(mask_pred, true_masks).item()
            # else:
                # pred = torch.sigmoid(mask_pred)
                # pred = (pred > 0.5).float()
                # tot += dice_coeff(pred, true_masks).item()
            # pbar.update()

    # net.train()
    # return tot / n_val

def eval_net(net, test_loader, device):
    """Evaluation"""
    net.eval()
    n_val = len(test_loader)  # the number of batch
    tot = 0
    mask_pix_acc = 0
    mask_dice = 0
    depth_metrics = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for data in test_loader:
          data['bg'] = data['bg'].to(device)
          data['image'] = data['image'].to(device)
          data['depth'] = data['depth'].to(device)
          data['mask'] = data['mask'].to(device)
          output_depth, output_mask = net(data)
		  
          #abs_diff = (output - target).abs()
          #self.mse = float((torch.pow((output - target).abs(), 2)).mean())
          depth_rmse = math.sqrt(float((torch.pow((output_depth - data['depth']).abs(), 2)).mean()))

          #criterion_depth = kornia.losses.SSIM(3, reduction='mean')
          criterion_mask = torch.nn.BCEWithLogitsLoss()
          with torch.no_grad():
            mask_pix_acc += iou(output_mask, data['mask'])
          mask_dice += criterion_mask(output_mask, data['mask']).item()
          depth_metrics += depth_rmse #criterion_depth(output_depth, data['depth']).item()
          tot += depth_rmse + criterion_mask(output_mask, data['mask']).item()
          pbar.update()

    net.train()
    tot /= n_val
    mask_pix_acc /= n_val
    mask_pix_acc = float(mask_pix_acc.cpu().numpy())
    mask_dice /= n_val
    depth_metrics /= n_val

    return tot, mask_pix_acc, mask_dice, depth_metrics

def saliency_iou(saliency_seg, saliency_depth, threshold):
    mask_seg = np.logical_not(saliency_seg < threshold)
    mask_depth = np.logical_not(saliency_depth < threshold)
    union = np.logical_or(mask_seg, mask_depth)
    inter = np.logical_and(mask_seg, mask_depth)
    iou = np.sum(inter) / np.sum(union)
    return iou


def calculate_overlap(img_seg, img_depth, bp_seg, bp_depth, img_resize_height=160, img_resize_width=160, sample_rate=20):
    pred_seg = bp_seg.forward(img_seg.to(device))  # predict lbl / depth: [h, w]
    pred_depth = bp_depth.forward(img_depth.to(device))

    img_iou1, img_iou2, img_iou3, img_iou4 = [], [], [], []

    y1, y2 = int(0.40810811 * img_resize_height), int(0.99189189 * img_resize_height)
    x1, x2 = int(0.03594771 * img_resize_width), int(0.96405229 * img_resize_width)
    total_pixel = 0

    for pos_i in tqdm(range(y1+sample_rate, y2, sample_rate)):
        for pos_j in tqdm(range(x1+sample_rate, x2, sample_rate)):
            bp_seg.backward(pos_i=pos_i, pos_j=pos_j, idx=pred_seg[pos_i, pos_j])
            bp_depth.backward(pos_i=pos_i, pos_j=pos_j, idx=pred_depth[pos_i, pos_j])
            _, output_saliency_seg = bp_seg.generate()  # output_saliency: [h, w]
            _, output_saliency_depth = bp_depth.generate()

            output_saliency_seg = output_saliency_seg[y1:y2, x1:x2]
            output_saliency_depth = output_saliency_depth[y1:y2, x1:x2]
            # normalized saliency map for a pixel in an image
            if np.max(output_saliency_seg) > 0:
                output_saliency_seg = (output_saliency_seg - np.min(output_saliency_seg)) / np.max(output_saliency_seg)
            if np.max(output_saliency_depth) > 0:
                output_saliency_depth = (output_saliency_depth - np.min(output_saliency_depth)) / np.max(output_saliency_depth)

            iou1 = saliency_iou(saliency_seg=output_saliency_seg, saliency_depth=output_saliency_depth, threshold=0.05)
            iou2 = saliency_iou(saliency_seg=output_saliency_seg, saliency_depth=output_saliency_depth, threshold=0.1)
            iou3 = saliency_iou(saliency_seg=output_saliency_seg, saliency_depth=output_saliency_depth, threshold=0.5)
            iou4 = saliency_iou(saliency_seg=output_saliency_seg, saliency_depth=output_saliency_depth, threshold=0.9)

            total_pixel += 1
            img_iou1.append(iou1)
            img_iou2.append(iou2)
            img_iou3.append(iou3)
            img_iou4.append(iou4)

    return img_iou1, img_iou2, img_iou3, img_iou4  # list, for all pixels evaluated

def evaluate(model, testloader, img_resize_height=160, img_resize_width=160, sample_rate=20):
    model.eval()
    bp_seg = BackPropagation(model=model, task="seg")
    bp_depth = BackPropagation(model=model, task="depth")
    result_img = []

    for i, data in enumerate(testloader):
        img_iou = calculate_overlap(img_seg=data['mask'], img_depth=data['depth'], bp_seg=bp_seg, bp_depth=bp_depth, img_resize_height=img_resize_height, img_resize_width=img_resize_width, sample_rate=sample_rate)
        result_img.append(img_iou)
    result_img_out = np.array(result_img, dtype=float)  # [num_image, num_metrics=4, num_pixels_for_each_image]
    # np.save("saliency_eval_pixel/{}_iou_try.npy".format(args.model_name), result_img_out)
    return result_img_out


def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        hist: confusion matrix.
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


# def per_class_pixel_accuracy(hist):
    # """Computes the average per-class pixel accuracy.
    # The per-class pixel accuracy is a more fine-grained
    # version of the overall pixel accuracy. A model could
    # score a relatively high overall pixel accuracy by
    # correctly predicting the dominant labels or areas
    # in the image whilst incorrectly predicting the
    # possibly more important/rare labels. Such a model
    # will score a low per-class pixel accuracy.
    # Args:
        # hist: confusion matrix.
    # Returns:
        # avg_per_class_acc: the average per-class pixel accuracy.
    # """
    # correct_per_class = torch.diag(hist)
    # total_per_class = hist.sum(dim=1)
    # per_class_acc = correct_per_class / (total_per_class + EPS)
    # avg_per_class_acc = nanmean(per_class_acc)
    # return avg_per_class_acc


# def jaccard_index(hist):
    # """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    # Args:
        # hist: confusion matrix.
    # Returns:
        # avg_jacc: the average per-class jaccard index.
    # """
    # A_inter_B = torch.diag(hist)
    # A = hist.sum(dim=1)
    # B = hist.sum(dim=0)
    # jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    # avg_jacc = nanmean(jaccard)
    # return avg_jacc


# def dice_coefficient(hist):
    # """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    # Args:
        # hist: confusion matrix.
    # Returns:
        # avg_dice: the average per-class dice coefficient.
    # """
    # A_inter_B = torch.diag(hist)
    # A = hist.sum(dim=1)
    # B = hist.sum(dim=0)
    # dice = (2 * A_inter_B) / (A + B + EPS)
    # avg_dice = nanmean(dice)
    # return avg_dice


def eval_metrics(true, pred, num_classes):
    """Computes various segmentation metrics on 2D feature maps.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.
    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    hist = torch.zeros((num_classes, num_classes))
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    #avg_per_class_acc = per_class_pixel_accuracy(hist)
    # avg_jacc = jaccard_index(hist)
    # avg_dice = dice_coefficient(hist)
    return overall_acc#, avg_per_class_acc, avg_jacc, avg_dice

def evaluate(model, rgb, depth, crop, batch_size=6):
    def compute_errors(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        
        a1 = (thresh < 1.25   ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()

        return a1, a2, a3, abs_rel, rmse, log_10

    depth_scores = np.zeros((6, len(rgb))) # six metrics

    bs = batch_size

    for i in range(len(rgb)//bs):    
        x = rgb[(i)*bs:(i+1)*bs,:,:,:]
        
        # Compute results
        true_y = depth[(i)*bs:(i+1)*bs,:,:]
        pred_y = scale_up(2, predict(model, x/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0
        
        # Test time augmentation: mirror image estimate
        pred_y_flip = scale_up(2, predict(model, x[...,::-1,:]/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y = pred_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y_flip = pred_y_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        
        # Compute errors per image in batch
        for j in range(len(true_y)):
            errors = compute_errors(true_y[j], (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j])))
            
            for k in range(len(errors)):
                depth_scores[k][(i*bs)+j] = errors[k]

    e = depth_scores.mean(axis=1)
    return e