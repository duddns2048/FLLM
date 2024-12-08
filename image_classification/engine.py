# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import time

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import kornia as K

from utils import DistillationLoss
import utils
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

def hook_fn(module, input, output):
    # feature_map = []
    global feature_map
    feature_map.append(output.detach().cpu())

def save_predictions(images, activation_map, save_dir='./', i=0, epoch=None):
    # os.makedirs(save_dir, exist_ok=True)

    # Convert tensors to numpy arrays for visualization
    images_np = images.cpu().detach().numpy().transpose(1, 2, 0)
    images_np = (images_np-images_np.min())/(images_np.max()-images_np.min()) *255
    images_np = images_np.astype('uint8')
    images_pil = Image.fromarray(images.cpu().detach().numpy().transpose(1, 2, 0).astype('uint8'))
    activation_map_np = activation_map
    output_np = (activation_map_np > 0.5)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(images_np)
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(activation_map_np, cmap='viridis') # 'hot'
    ax[1].set_title('feature activation')
    ax[1].axis('off')

    ax[2].imshow(output_np, cmap='gray') # 'jet' | 'tab20'
    ax[2].set_title('f_act mask')
    ax[2].axis('off')

    save_path = os.path.join(save_dir, f'f_act_e{epoch}_{i}.png')
    plt.savefig(save_path, bbox_inches='tight')
    # plt.savefig(save_dir, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def PGDAttack(x, y, model, attack_epsilon, attack_alpha, lower_limit, loss_fn, upper_limit, max_iters, random_init):
    model.eval()

    delta = torch.zeros_like(x).cuda()
    if random_init:
        for iiiii in range(len(attack_epsilon)):
            delta[:, iiiii, :, :].uniform_(-attack_epsilon[iiiii][0][0].item(), attack_epsilon[iiiii][0][0].item())
    
    adv_imgs = clamp(x+delta, lower_limit, upper_limit)
    max_iters = int(max_iters)
    adv_imgs.requires_grad = True 

    with torch.enable_grad():
        for _iter in range(max_iters):
            
            outputs = model(adv_imgs)

            loss = loss_fn(outputs, y)

            grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None, 
                    only_inputs=True)[0]

            adv_imgs.data += attack_alpha * torch.sign(grads.data) 
            
            adv_imgs = clamp(adv_imgs, x-attack_epsilon, x+attack_epsilon)

            adv_imgs = clamp(adv_imgs, lower_limit, upper_limit)

    return adv_imgs.detach()

def patch_level_aug(input1, patch_transform, upper_limit, lower_limit):
    bs, channle_size, H, W = input1.shape
    patches = input1.unfold(2, 16, 16).unfold(3, 16, 16).permute(0,2,3,1,4,5).contiguous().reshape(-1, channle_size,16,16)
    patches = patch_transform(patches)
 
    patches = patches.reshape(bs, -1, channle_size,16,16).permute(0,2,3,4,1).contiguous().reshape(bs, channle_size*16*16, -1)
    output_images = F.fold(patches, (H,W), 16, stride=16)
    output_images = clamp(output_images, lower_limit, upper_limit)
    return output_images


def train_one_epoch(args, model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, optimizer_parameters,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, use_wandb=False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter=" | ", use_wandb=use_wandb)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
    mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)

    i = 0
    # for samples, targets in tqdm(data_loader):
    #     samples = samples.to(device, non_blocking=True)
    #     targets = targets.to(device, non_blocking=True)
        
        
    #     optimizer.zero_grad()
    #     outputs = model(samples)
    #     loss = criterion(samples, outputs, targets)

    #     loss_value += loss.item()
        
    #     if not math.isfinite(loss_value):
    #         print(f"Loss is {loss_value}, stopping training"))
    #         sys.exit(1)
            
    #     loss.backward()
    #     optimizer.step()
            
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # if i % 100 == 0:
        #     model.module.update_L(lam=0.1)
        # i = i + 1

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None: # default True
            samples, targets = mixup_fn(samples, targets)

        if args.use_patch_aug: # default False
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(16,16), scale=(0.85,1.0), ratio=(1.0,1.0), p=0.1),
                K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.1),
                K.augmentation.RandomHorizontalFlip(p=0.1)
                )
            aug_samples = patch_level_aug(samples, patch_transform, upper_limit, lower_limit)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        with torch.cuda.amp.autocast():
            if args.use_patch_aug: # default False
                outputs2 = model(aug_samples)
                loss = criterion(aug_samples, outputs2, targets)
                if args.recon_loss > 1e-4:
                    loss = loss + args.recon_loss * model.module.recon_loss()
                loss_scaler._scaler.scale(loss).backward(create_graph=is_second_order)
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)
            else:
                outputs, feature_map = model(samples)
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        
        # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = samples.shape[0]
        

        torch.cuda.synchronize()
        if model_ema is not None: # default True
            model_ema.update(model)

        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        

        # normalize the FFT kernel
        # model.module.normalize_parameters()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, output_dir, mask=None, adv=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    i=0
    # switch to evaluation mode
    model.eval()
    
    # feature_map = []
    # model.module.norm.register_forward_hook(hook_fn)

    os.makedirs(os.path.join(output_dir,'./feature_activation'), exist_ok=True)
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        i+=1

        if adv == 'FGSM':
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (1 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=1, random_init=False)
        elif adv == "PGD":
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (0.5 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input)
            else:
                output, feature_map = model(images)
            loss = criterion(output, target)
            
        # visual_tokens = feature_map[:, 1:, :].cpu().detach().numpy()  # Exclude [CLS] token
        # avg_activations = np.mean(visual_tokens, axis=-1)
        # activation_map = avg_activations.reshape(-1, 14, 14)
        
        if epoch%10==0:
            if ((i % 50)<5) and i<500: # i must smaller than batch size
                j= i % 50
                visual_tokens = feature_map[j, 1:, :].cpu().detach().numpy()  # Exclude [CLS] token
                avg_token_feature = np.mean(visual_tokens, axis=0)
                # activation = (visual_tokens - avg_token_feature).norm(dim=-1)
                activation = np.linalg.norm(visual_tokens - avg_token_feature, axis=-1)

                mag_min, mag_max = activation.min(), activation.max()
                mag_activation = (activation - mag_min) / (mag_max - mag_min)
                activation_map = mag_activation.reshape(14, 14)
                save_predictions(images[j], activation_map, save_dir=os.path.join(output_dir,'./feature_activation'), i=i, epoch=epoch)

        

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('================== Acc@1 {top1.global_avg:.3f} | Acc@5 {top5.global_avg:.3f} | loss {losses.global_avg:.3f}==================='
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
