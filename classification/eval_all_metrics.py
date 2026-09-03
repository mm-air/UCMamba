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
from utils.utils import NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema, load_pretrained_revise
from loss_SeqCon import SeqCon

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, auc, roc_curve, roc_auc_score, \
    accuracy_score, precision_recall_fscore_support
from statistics import mean
from torchvision import datasets, transforms
from data.samplers import SubsetRandomSampler
from timm.data import Mixup

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
    parser.add_argument('--eval_model',
                        help='load weight from checkpoint of the model for evaluation')

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


def get_test_results_regression(data_loader, model, device, args):
    'Ref: https://github.com/GorkemP/labeled-images-for-ulcerative-colitis'
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # y_true.append(target.tolist()[0])
            y_true = y_true + target.tolist()
            if args.use_ctr:
                output, _ = model(data)
            else:
                output = model(data)
            prediction = torch.clip(torch.round(output), 0, 3)
            y_pred = y_pred + [prediction[i][0].item() for i in range(len(prediction))]

    return y_true, y_pred


def get_mean_sensitivity_specificity(y_true: list, y_pred: list):
    'Ref: https://github.com/GorkemP/labeled-images-for-ulcerative-colitis'
    cm_all = confusion_matrix(y_true, y_pred)
    cr_all = classification_report(y_true, y_pred, output_dict=True)

    specificities = []
    for i in range(cm_all.shape[0]):
        total = np.sum(cm_all)
        tp = cm_all[i][i]
        fn = np.sum(cm_all[i, :]) - tp
        fp = np.sum(cm_all[:, i]) - tp
        tn = total - (tp + fp + fn)
        specificity = tn / (tn + fp)
        specificities.append(specificity)

    return cr_all["macro avg"]["recall"], mean(specificities)


def show_evaluate_metrics(test_loader, model, device, args):
    'Ref: https://github.com/GorkemP/labeled-images-for-ulcerative-colitis'
    y_true, y_pred = get_test_results_regression(test_loader, model, device, args)
    accuracy = accuracy_score(y_true, y_pred)
    QWK_kappa_score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm_all = confusion_matrix(y_true, y_pred, normalize="true")  # confusion matrix
    mean_sensitivity, mean_specificity = get_mean_sensitivity_specificity(y_true, y_pred)

    prf1_4classes = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    macro_precisions = prf1_4classes[0].mean()
    macro_recalls = prf1_4classes[1].mean()
    macro_f1s = prf1_4classes[2].mean()
    return accuracy, macro_f1s, mean_sensitivity, mean_specificity, QWK_kappa_score


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

from PIL import Image, ImageDraw
class FixedRotate:
    def __init__(self, angle, fill=(0, 0, 0)):
        self.angle = angle
        self.fill = fill

    def __call__(self, img):
        return transforms.functional.rotate(
            img, angle=self.angle, interpolation=Image.BILINEAR, fill=self.fill
        )

class CenterCropRatio:
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, img):
        w, h = img.size
        new_w = int(w * self.ratio)
        new_h = int(h * self.ratio)
        img = transforms.functional.center_crop(img, (new_h, new_w))
        return img

def build_loader_uc(config, aug_mode=False):
    # config.defrost()
    # dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    # config.freeze()
    data_transform = {
        "train": transforms.Compose([  # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-180, 180)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([#FixedRotate(60),
                                    #CenterCropRatio(0.6),
                                    transforms.Resize((224, 224)),
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
    # dataset_val, _ = build_dataset(is_train=False, config=config)
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
    # maxk = max(topk)
    batch_size = target.size(0)
    # _, pred = output.topk(maxk, 1, True, True)
    # pred = pred.t()
    # correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    pred = torch.round(output)
    acc_per_sample = pred.eq(target.view_as(pred)).sum() * 100. / batch_size
    return acc_per_sample


def main(config, args):
    AUG_MODE = True  # Contrastive learning not use augmentation
    if args.use_ctr == False:
        AUG_MODE = False
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = build_loader_uc(config, aug_mode=AUG_MODE)
    #dataset_test, data_loader_test = build_loader_wholeDataset(config)

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
            logger.info(flop_count_str(FlopCountAnalysis(model, (dataset_test[0][0][None],))))
    torch.cuda.empty_cache()
    dist.barrier()
    model.cuda()


    model_without_ddp = model
    model_ema = None
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        if args.ssm_forwardtype == 'v05hvs_noz': #6D
            load_pretrained_revise(config, model_without_ddp, logger, model_ema, num_dir=6)
        elif args.ssm_forwardtype in ('v05_noz', 'v05hv_noz', 'v05sv_noz', 'v05hs_noz'):
            load_pretrained_ema(config, model_without_ddp, logger, model_ema)
        elif args.ssm_forwardtype in ('v05h_noz', 'v05v_noz', 'v05s_noz'): #6D
            load_pretrained_revise(config, model_without_ddp, logger, model_ema, num_dir=2)

    #model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
    logger.info(f"==============> Loading weight {args.eval_model} for validation......")
    ckpt = torch.load(args.eval_model, map_location='cpu')
    model.load_state_dict(ckpt["model"])
    model.eval()
    ## print all evaluation metrics
    logger.info('start the evaluation')
    accuracy, macro_f1s, mean_sensitivity, mean_specificity, QWK_kappa_score = show_evaluate_metrics(
                data_loader_test, model, device='cuda', args=args)
    logger.info("Average Accuracy; macro_f1s; Average Sensitivity; Average Specificity; QWK score")
    logger.info("{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f}".format(accuracy, macro_f1s, mean_sensitivity, mean_specificity, QWK_kappa_score))

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
        if aug_mode:
            images = torch.cat([images[0], images[1]], dim=0)  # 2bs
            target = target.repeat(2)  # 2bs
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

    seed = config.SEED + dist.get_rank()
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
        # linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        # linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
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
