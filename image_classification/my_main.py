import os
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import models.vit
import models.vit_llama
from datasets import build_dataset, build_transform
from engine import train_one_epoch, evaluate
from utils import DistillationLoss
from utils import RASampler
from args import get_args_parser
import utils
from optimizer_utils import my_create_optimizer
import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot_graph(save_dir, train_losses, valid_losses, plot):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label=f'Train {plot}')
    plt.plot(valid_losses, label=f'Valid {plot}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{plot}')
    plt.title(f'Training and Validation {plot} per Epoch')
    plt.legend()
    plt.grid(True)

    # 그래프를 PNG 파일로 저장
    plt.savefig(save_dir+f'/{plot}_graph.png')
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name,  'train'), exist_ok=True)
    output_dir = os.path.join(args.output_dir, args.exp_name, 'train')
    
    # save args
    args_dict = vars(args)
    with open(os.path.join(output_dir,'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
    
    # save command order
    command_file_name = './train_command.txt' if not args.eval else './test_command.txt'
    command_line_arguments = ' '.join(sys.argv)
    with open(os.path.join(output_dir, command_file_name), 'w') as f:
        f.write(command_line_arguments)
    
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active: # default True
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None
    )
    
    if 'llama' in args.model:
        print("Loading LLaMA checkpoints")
        start_time = time.time()
        checkpoints = sorted(Path(args.llama_path).glob("*.pth"))
        ckpt_path = checkpoints[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.llama.custom_load_state_dict(checkpoint, tail=True, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds") 
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  
    model.to(device)
    
    model_ema = None
    if args.model_ema: # default True
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer, optimizer_parameters = my_create_optimizer(args, model, param_filter=args.param_filter)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()
    if args.mixup > 0.: # default True
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    teacher_model = None
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    
    train_loss_list=[]
    train_acc1_list=[]
    train_acc5_list=[]
    test_loss_list=[]
    test_acc1_list=[]
    test_acc5_list=[]
     
    if args.resume and os.path.isfile(args.resume): # default False
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            df = pd.read_csv(f'{output_dir}/train_log.csv')
            train_loss_list = list(df['train_loss'])
            test_loss_list=list(df['test_loss'])
            test_acc1_list=list(df['acc1'])
            test_acc5_list=list(df['acc5'])
    
    
    start_time = time.time()
    max_accuracy = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            args, model, criterion, data_loader_train,
            optimizer, optimizer_parameters, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning,
            use_wandb=args.wandb
        )

        lr_scheduler.step(epoch)
        train_loss_list.append(train_stats['loss'])
        # train_acc1_list.append(train_stats['acc1'])
        # train_acc5_list.append(train_stats['acc5'])
        
        if (epoch+1) % args.test_every == 0:
            if args.output_dir:
                checkpoint_paths = [os.path.join(output_dir, 'checkpoint.pth')]
                for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'model_ema': get_state_dict(model_ema),
                            'scaler': loss_scaler.state_dict(),
                            'args': args,
                        }, checkpoint_path)
                        

            test_stats = evaluate(data_loader_val, model, device, epoch, output_dir)
            test_loss_list.append(test_stats['loss'])
            test_acc1_list.append(test_stats['acc1'])
            test_acc5_list.append(test_stats['acc5'])
            
            print(f"============= Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}% =============")
            if test_stats["acc1"] > max_accuracy:
                checkpoint_paths = [os.path.join(output_dir, 'best_model.pth')]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'================== Max accuracy: {max_accuracy:.2f}% ==================')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with open(os.path.join(output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                    
        if args.save_train_log_dataframe:
            epoch_data = {
                "epoch": list(range(1, epoch + 2)),
                "acc1" : test_acc1_list,
                "acc5" : test_acc5_list,
                "train_loss" : train_loss_list,
                "test_loss" : test_loss_list,
            }
            df = pd.DataFrame(epoch_data)
            csv_path = os.path.join(output_dir, 'train_log.csv')
            df.to_csv(csv_path, index=False)
            
        plot_graph(output_dir, train_loss_list, test_loss_list,'loss')
        plot_graph(output_dir, test_acc1_list, test_acc1_list,'acc1')
        plot_graph(output_dir, test_acc5_list, test_acc5_list,'acc5')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))