# --------------------------------------------------------
# Modified by $@#Anonymous#@$
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.byoanet import halonet_h1
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import  NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema, load_pretrained_revise
from loss_SeqCon import SeqCon

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma

import matplotlib.pyplot as plt

if torch.multiprocessing.get_start_method() != "spawn":
    print(f"||{torch.multiprocessing.get_start_method()}||", end="")
    torch.multiprocessing.set_start_method("spawn", force=True)

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default="", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # Add by Shirley
    parser.add_argument('--loss_ctr', type=str, default="SeqCon", help="loss_ctr")
    parser.add_argument("--temper", type=float, default=0.2, help="temper")
    parser.add_argument("--weight_ctr", "--w", type=float, default=0.1, help="weight_ctr")
    parser.add_argument('--lr', type=float, default=5e-3, help="learning rate")
    parser.add_argument('--use_ctr', type=str2bool, default=False, help="use contrastive learning")
    parser.add_argument('--epochs', default=100, type=int, help="number of epoch")
    parser.add_argument('--ssm_forwardtype', type=str, default="v05_hvs", help="SSM_FORWARDTYPE")
    parser.add_argument('--seed', default=0, type=int, help="number of epoch")
    parser.add_argument('--ctr_aug', type=str2bool, default=True, help="use contrastive learning")

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="/dataset/ImageNet_ILSVRC2012", help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

from torchvision import datasets, transforms
from data.samplers import SubsetRandomSampler
from timm.data import Mixup
class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def build_loader_uc(config, aug_mode=False):
    data_transform = {
        "train": transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation((-180, 180)),
                                     transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    }
    if aug_mode:
        dataset_train = datasets.ImageFolder(root=os.path.join(config.DATA.DATA_PATH, 'train'),
                                             transform=TwoCropTransform(data_transform["train"]))
        dataset_val = datasets.ImageFolder(root=os.path.join(config.DATA.DATA_PATH, 'val'),
                                           transform=data_transform["val"])
        dataset_test = datasets.ImageFolder(root=os.path.join(config.DATA.DATA_PATH, 'test'),
                                            transform=data_transform["test"])
    else:
        dataset_train = datasets.ImageFolder(root=os.path.join(config.DATA.DATA_PATH, 'train'),
                                         transform=data_transform["train"])
        dataset_val = datasets.ImageFolder(root=os.path.join(config.DATA.DATA_PATH, 'val'),
                                            transform=data_transform["val"])
        dataset_test = datasets.ImageFolder(root=os.path.join(config.DATA.DATA_PATH, 'test'),
                                            transform=data_transform["test"])
    print(f"rank {dist.get_rank()} successfully build train dataset")
    #dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"rank {dist.get_rank()} successfully build val dataset")
    print(f"rank {dist.get_rank()} successfully build test dataset")
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )
        sampler_test = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn


class AverageMeter:
    """Computes and stores the average and current value"""
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


def accuracy_reg(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    #maxk = max(topk)
    batch_size = target.size(0)
    #_, pred = output.topk(maxk, 1, True, True)
    #pred = pred.t()
    #correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    pred = torch.round(output)
    acc_per_sample = pred.eq(target.view_as(pred)).sum() * 100. / batch_size
    return acc_per_sample


def main(config, args):
    AUG_MODE = args.ctr_aug   #Contrastive learning not use augmentation
    if args.use_ctr == False:
        AUG_MODE = False
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = build_loader_uc(config, aug_mode=AUG_MODE)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    if dist.get_rank() == 0:
        if hasattr(model, 'flops'):
            logger.info(str(model))
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"number of params: {n_parameters}")
            flops = model.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")
        else:
            logger.info(flop_count_str(FlopCountAnalysis(model, (dataset_val[0][0][None],))))
    torch.cuda.empty_cache()
    dist.barrier()
    model.cuda()
    model_without_ddp = model

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)


    optimizer = build_optimizer(config, model, logger)
    model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    ########### Contrastive representation learning. Add by Shirley
    if config.MODEL.NUM_CLASSES == 1:
        criterion = torch.nn.MSELoss()  #Regression
    if args.loss_ctr == 'SeqCon':
        norm_val = 2 / (len(data_loader_train.dataset))
        criterion_ctr = SeqCon(temperature=args.temper, norm_val=norm_val, feature_sim='l2')
    else:
        criterion_ctr = None
        print("incorrect selection for contr")

    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    val_max_accuracy = 0.0; tst_max_accuracy = 0.0; val_max_corrTst = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        val_max_accuracy, val_max_corrTst, tst_max_accuracy, max_accuracy_ema = load_checkpoint_ema(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger, model_ema)
        if args.use_ctr:
            val_acc1, val_loss, val_loss_reg, val_loss_ctr = validate_ctr(config, data_loader_val, model, criterion, criterion_ctr, args.weight_ctr, aug_mode=AUG_MODE)
        else:
            val_acc1, val_loss = validate(config, data_loader_val, model, criterion)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {val_acc1:.1f}%")
        logger.info(f'Max val accuracy: {val_max_accuracy:.2f}%; correspondant test accuracy:  {val_max_corrTst:.2f}%;')
        logger.info(f'Test Max accuracy: {tst_max_accuracy:.2f}%')
        if model_ema is not None:
            acc1_ema, loss_ema = validate(config, data_loader_val, model_ema.ema, criterion)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")

        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        if args.ssm_forwardtype == 'v05hvs_noz': #6D
            load_pretrained_revise(config, model_without_ddp, logger, model_ema, num_dir=6)
        elif args.ssm_forwardtype in ('v05_noz', 'v05hv_noz', 'v05sv_noz', 'v05hs_noz'): #4D
            load_pretrained_ema(config, model_without_ddp, logger, model_ema)
        elif args.ssm_forwardtype in ('v05h_noz', 'v05v_noz', 'v05s_noz'):   #2D
            load_pretrained_revise(config, model_without_ddp, logger, model_ema, num_dir=2)
        elif args.ssm_forwardtype in ('v05s1d_noz', 'v05h1d_noz', 'v05v1d_noz'): #1D
            load_pretrained_revise(config, model_without_ddp, logger, model_ema, num_dir=1)
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
        
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE and (dist.get_rank() == 0):
        logger.info(f"throughput mode ==============================")
        throughput(data_loader_val, model, logger)
        if model_ema is not None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            throughput(data_loader_val, model_ema.ema, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    hist = {
        'train_acc': [], 'val_acc': [], 'test_acc': [],
        'train_loss': [], 'val_loss': [], 'test_loss': []
    }
    # Only track contrastive losses if applicable
    if args.use_ctr:
        hist.update({
            'train_loss_reg': [], 'val_loss_reg': [], 'test_loss_reg': [],
            'train_loss_ctr': [], 'val_loss_ctr': [], 'test_loss_ctr': []
        })
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        ####### Train
        if args.use_ctr:
            trn_acc1, trn_loss, trn_loss_reg, trn_loss_ctr = train_ctr(config, model, criterion, criterion_ctr, args.weight_ctr, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema, aug_mode=AUG_MODE)
        else:
            trn_acc1, trn_loss = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema)
        if dist.get_rank() == 0 and (epoch % 50 == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint_ema(config, epoch, model_without_ddp, val_max_accuracy, val_max_corrTst, tst_max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema)

        ####### Validation
        if args.use_ctr:
            val_acc1, val_loss, val_loss_reg, val_loss_ctr = validate_ctr(config, data_loader_val, model, criterion, criterion_ctr, args.weight_ctr, aug_mode=AUG_MODE)
            tst_acc1, tst_loss, tst_loss_reg, tst_loss_ctr = validate_ctr(config, data_loader_test, model, criterion, criterion_ctr, args.weight_ctr, aug_mode=AUG_MODE)
        else:
            val_acc1, val_loss = validate(config, data_loader_val, model, criterion)
            tst_acc1, tst_loss = validate(config, data_loader_test, model, criterion)

        if val_max_accuracy < val_acc1:
            val_max_accuracy = val_acc1
            val_max_corrTst = tst_acc1
            save_checkpoint_ema(config, epoch, model_without_ddp, val_max_accuracy, val_max_corrTst, tst_max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema, save_name='val_best_model')
        if tst_max_accuracy < tst_acc1:
            tst_max_accuracy = tst_acc1
            save_checkpoint_ema(config, epoch, model_without_ddp, val_max_accuracy, val_max_corrTst, tst_max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema, save_name='tst_best_model')
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
            max_accuracy_ema = max(max_accuracy_ema, acc1_ema)
            logger.info(f'Max accuracy ema: {max_accuracy_ema:.2f}%')
        logger.info(f"Accuracy of the network on the {len(dataset_val)} val images: {val_acc1:.1f}%")
        logger.info(f'Max val accuracy: {val_max_accuracy:.2f}%; correspondant test accuracy:  {val_max_corrTst:.2f}%;')
        logger.info(f"Accuracy of the network on the {len(dataset_test)} test images: {tst_acc1:.1f}%")
        logger.info(f'Test Max accuracy: {tst_max_accuracy:.2f}%')

        # Append results
        hist['train_acc'].append(trn_acc1); hist['val_acc'].append(val_acc1); hist['test_acc'].append(tst_acc1)
        hist['train_loss'].append(trn_loss); hist['val_loss'].append(val_loss); hist['test_loss'].append(tst_loss)
        if args.use_ctr:
            hist['train_loss_reg'].append(trn_loss_reg); hist['val_loss_reg'].append(val_loss_reg); hist['test_loss_reg'].append(tst_loss_reg)
            hist['train_loss_ctr'].append(trn_loss_ctr); hist['val_loss_ctr'].append(val_loss_ctr); hist['test_loss_ctr'].append(tst_loss_ctr)

    # Timing
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')

    result_file = os.path.join(config.OUTPUT, 'Res.txt')
    with open(result_file, 'a') as output_save:
        output_save.write(f'\nTraining time {total_time_str}\n')
        for k in ['train_acc', 'val_acc', 'test_acc', 'train_loss', 'val_loss', 'test_loss']:
            output_save.write(f'{k.replace("_", " ").title()} of all epochs: {hist[k]}\n')
        if args.use_ctr:
            for k in ['train_loss_reg', 'val_loss_reg', 'test_loss_reg','train_loss_ctr', 'val_loss_ctr', 'test_loss_ctr']:
                output_save.write(f'{k.replace("_", " ").title()} of all epochs: {hist[k]}\n')
        best_val_idx = np.argmax(hist['val_acc'])
        best_test_idx = np.argmax(hist['test_acc'])
        best_val_epoch = config.TRAIN.START_EPOCH + best_val_idx
        best_test_epoch = config.TRAIN.START_EPOCH + best_test_idx
        output_save.write(f'\nResults of best val model in epoch {best_val_epoch}/{len(hist["val_acc"])}:\n')
        output_save.write(f'Val acc: {hist["val_acc"][best_val_idx]:.4f}\n')
        output_save.write(f'Test acc: {hist["test_acc"][best_val_idx]:.4f}\n')
        output_save.write(f'Best test acc: {hist["test_acc"][best_test_idx]:.4f} in epoch {best_test_epoch}\n')
        output_save.write(f'\nResults of best test model in epoch {best_test_epoch}/{len(hist["test_acc"])}:\n')
        if args.use_ctr:
            output_save.write(f'loss_ctr_type: {args.loss_ctr}, weight_ctr: {args.weight_ctr}, temperature: {args.temper}\n')

    # Plot results
    def plot_metric(keys, filename):
        plt.figure()
        for k in keys:
            plt.plot(hist[k], label=k.split('_')[0])
        plt.legend()
        plt.savefig(os.path.join(config.OUTPUT, filename))

    plot_metric(['train_acc', 'val_acc', 'test_acc'], 'Acc.png')
    plot_metric(['train_loss', 'val_loss', 'test_loss'], 'loss.png')
    if args.use_ctr:
        plot_metric(['train_loss_reg', 'val_loss_reg', 'test_loss_reg'], 'loss_reg.png')
        plot_metric(['train_loss_ctr', 'val_loss_ctr', 'test_loss_ctr'], 'loss_ctr.png')


def train_ctr(config, model, criterion, criterion_ctr, weight_ctr, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None, model_time_warmup=50, aug_mode=False):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    loss_ctr_meter = AverageMeter()
    loss_reg_meter = AverageMeter()
    acc1_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        torch.cuda.reset_peak_memory_stats()
        if aug_mode:
            samples = torch.cat([samples[0], samples[1]], dim=0)  # 2bs
            targets = targets.repeat(2)  # 2bs, class mode
        targets = targets.unsqueeze(1).to(torch.float16)### classfication task -> Regression task, need to revise this
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs, features = model(samples)
        loss_reg = criterion(outputs, targets)
        loss_ctr = criterion_ctr(features, targets)
        loss = loss_reg + weight_ctr * loss_ctr
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        loss_reg_meter.update(loss_reg.item(), targets.size(0))
        loss_ctr_meter.update(loss_ctr.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        acc1 = accuracy_reg(outputs, targets)
        acc1_meter.update(acc1.item(), targets.size(0))

        if idx > model_time_warmup:
            model_time.update(batch_time.val - data_time.val)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'model time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'class loss {loss_reg_meter.val:.4f} ({loss_reg_meter.avg:.4f})\t'
                f'contrast loss {loss_ctr_meter.val:.4f} ({loss_ctr_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return acc1_meter.avg, loss_meter.avg, loss_reg_meter.avg, loss_ctr_meter.avg


@torch.no_grad()
def validate_ctr(config, data_loader, model, criterion, criterion_ctr, weight_ctr, aug_mode=False):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_reg_meter = AverageMeter()
    loss_ctr_meter = AverageMeter()
    acc1_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        target = target.unsqueeze(1).to(torch.float16)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output, features = model(images)

        # measure accuracy and record loss
        loss_reg = criterion(output, target)
        loss_ctr = criterion_ctr(output, target)
        loss = loss_reg + loss_ctr * weight_ctr
        acc1 = accuracy_reg(output, target)
        acc1 = reduce_tensor(acc1)
        acc1_meter.update(acc1.item(), target.size(0))
        loss = reduce_tensor(loss)
        loss_reg = reduce_tensor(loss_reg)
        loss_meter.update(loss.item(), target.size(0))
        loss_reg_meter.update(loss_reg.item(), target.size(0))
        loss_ctr_meter.update(loss_ctr.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Loss {loss_reg_meter.val:.4f} ({loss_reg_meter.avg:.4f})\t'
                f'Loss {loss_ctr_meter.val:.4f} ({loss_ctr_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')
    return acc1_meter.avg, loss_meter.avg, loss_reg_meter.avg, loss_ctr_meter.avg


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None, model_time_warmup=50):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    acc1_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        torch.cuda.reset_peak_memory_stats()
        samples = samples.cuda(non_blocking=True)
        targets = targets.unsqueeze(1).to(torch.float16)  ### classfication task -> Regression task, need to revise this
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        acc1 = accuracy_reg(outputs, targets)
        acc1_meter.update(acc1.item(), targets.size(0))

        if idx > model_time_warmup:
            model_time.update(batch_time.val - data_time.val)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'model time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return acc1_meter.avg, loss_meter.avg


@torch.no_grad()
def validate(config, data_loader, model, criterion):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    #acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.unsqueeze(1).to(torch.float16)  ### classfication task -> Regression task, need to revise this
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #acc1, acc5 = accuracy(output, target, topk=(1, 2))

        loss = reduce_tensor(loss)
        loss_meter.update(loss.item(), target.size(0))
        #acc1 = reduce_tensor(acc1)
        #acc5 = reduce_tensor(acc5)
        #acc1_meter.update(acc1.item(), target.size(0))
        #acc5_meter.update(acc5.item(), target.size(0))
        acc1 = accuracy_reg(output, target)
        acc1 = reduce_tensor(acc1)
        acc1_meter.update(acc1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                #f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')
    #return acc1_meter.avg, acc5_meter.avg, loss_meter.avg
    return acc1_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()

    #seed = config.SEED + dist.get_rank()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if True: 
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        #linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        #linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_lr * 1e-3
        linear_scaled_min_lr = linear_scaled_min_lr * 1e-2
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # to make sure all the config.OUTPUT are the same
    config.defrost()
    if dist.get_rank() == 0:
        obj = [config.OUTPUT]
        # obj = [str(random.randint(0, 100))] # for test
    else:
        obj = [None]
    dist.broadcast_object_list(obj)
    dist.barrier()
    config.OUTPUT = obj[0]
    print(config.OUTPUT, flush=True)
    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    if args.memory_limit_rate > 0 and args.memory_limit_rate < 1:
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_rate)
        usable_memory = torch.cuda.get_device_properties(0).total_memory * args.memory_limit_rate / 1e6
        print(f"===========> GPU memory is limited to {usable_memory}MB", flush=True)

    main(config, args)
