# --------------------------------------------------------
# Modified By $@#Anonymous#@$
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
from math import inf
import torch
import torch.distributed as dist
from timm.utils import ModelEma as ModelEma


def load_checkpoint_ema(config, model, optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(f"resuming model: {msg}")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.RESUME}! ")

    if model_ema is not None:
        if 'model_ema' in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=False)
            logger.info(f"resuming model_ema: {msg}")
        else:
            logger.warning(f"No 'model_ema' found in {config.MODEL.RESUME}! ")

    val_max_accuracy = 0.0; val_max_corrTst = 0.0; tst_max_accuracy = 0.0;
    max_accuracy_ema = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'val_max_accuracy' in checkpoint:
            val_max_accuracy = checkpoint['val_max_accuracy']
        if 'tst_max_accuracy' in checkpoint:
            tst_max_accuracy = checkpoint['tst_max_accuracy']
        if 'val_max_corrTst' in checkpoint:
            val_max_corrTst = checkpoint['val_max_corrTst']
        if 'max_accuracy_ema' in checkpoint:
            max_accuracy_ema = checkpoint['max_accuracy_ema']

    del checkpoint
    torch.cuda.empty_cache()
    return val_max_accuracy, val_max_corrTst, tst_max_accuracy, max_accuracy_ema


def revise_weights_6d_previous(checkpoint, kgroup, num_class=4):
    model_dict = checkpoint['model']
    # device = 'cuda:0'
    device = 'cpu'
    num = kgroup - 4
    k_multiply = [1, 2, 4, 8]  # A_logs and Ds should be revised for different layers
    if num!=0:
        for i in range(4):  # i: index of layer
            if i == 2:  # Layer 2 has 8 blocks and other layers have 2 blocks
                N = 8
            else:
                N = 2
            for j in range(N):  # j: index of block
                # op.x_proj_weight
                params_exist = model_dict['layers.%d.blocks.%d.op.x_proj_weight' % (i, j)]
                # params_add = torch.zeros([2,8,192]).to(device, dtype=torch.float32)
                #params_add = torch.zeros([2, *params_exist.shape[1:]]).to(device, dtype=torch.float32)
                params_add = params_exist[0:num]
                params_new = torch.concat([params_exist, params_add], dim=0)
                model_dict['layers.%d.blocks.%d.op.x_proj_weight' % (i, j)] = params_new
                # op.dt_projs_weight
                params_exist = model_dict['layers.%d.blocks.%d.op.dt_projs_weight' % (i, j)]
                #params_add = torch.zeros([2, *params_exist.shape[1:]]).to(device, dtype=torch.float32)
                params_add = params_exist[0:num]
                params_new = torch.concat([params_exist, params_add], dim=0)
                model_dict['layers.%d.blocks.%d.op.dt_projs_weight' % (i, j)] = params_new
                # op.dt_projs_bias
                params_exist = model_dict['layers.%d.blocks.%d.op.dt_projs_bias' % (i, j)]
                #params_add = torch.zeros([2, *params_exist.shape[1:]]).to(device, dtype=torch.float32)
                params_add = params_exist[0:num]
                params_new = torch.concat([params_exist, params_add], dim=0)
                model_dict['layers.%d.blocks.%d.op.dt_projs_bias' % (i, j)] = params_new
                # op.A_logs
                params_exist = model_dict['layers.%d.blocks.%d.op.A_logs' % (i, j)]
                #params_add = torch.zeros([384 * k_multiply[i]]).to(device, dtype=torch.float32)
                params_add = params_exist[0:96 * num * k_multiply[i]]
                params_new = torch.concat([params_exist, params_add], dim=0)
                model_dict['layers.%d.blocks.%d.op.A_logs' % (i, j)] = params_new
                # op.op.Ds
                params_exist = model_dict['layers.%d.blocks.%d.op.Ds' % (i, j)]
                #params_add = torch.zeros([384 * k_multiply[i]]).to(device, dtype=torch.float32)
                params_add = params_exist[0:96 * num * k_multiply[i]]
                params_new = torch.concat([params_exist, params_add], dim=0)
                model_dict['layers.%d.blocks.%d.op.Ds' % (i, j)] = params_new

    # classifier.head.weight
    #model_dict['classifier.head.weight'] = torch.zeros([4, 768]).to(device, dtype=torch.float32)
    model_dict['classifier.head.weight'] = model_dict['classifier.head.weight'][0:num_class]
    # classifier.head.bias
    #model_dict['classifier.head.bias'] = torch.zeros([4]).to(device, dtype=torch.float32)
    model_dict['classifier.head.bias'] = model_dict['classifier.head.bias'][0:num_class]
    return checkpoint


def revise_weights_6d_cornal_corn(checkpoint, num_class=4):
    import torch  # Ensure torch is imported
    model_dict = checkpoint['model']
    device = 'cpu'
    num = 2
    k_multiply = [1, 2, 4, 8]
    layers_blocks = [2, 2, 8, 2]  # Number of blocks for each layer
    param_keys = [
        'op.x_proj_weight',
        'op.dt_projs_weight',
        'op.dt_projs_bias',
        'op.A_logs',
        'op.Ds',
    ]
    for i, num_blocks in enumerate(layers_blocks):
        for j in range(num_blocks):
            for key in param_keys:
                pkey = f'layers.{i}.blocks.{j}.{key}'
                params_exist = model_dict[pkey]
                if key in ['op.A_logs', 'op.Ds']:
                    add_len = 96 * num * k_multiply[i]
                    params_add = params_exist[:add_len]
                else:
                    params_add = params_exist[:num]
                params_new = torch.concat([params_exist, params_add], dim=0)
                model_dict[pkey] = params_new

    model_dict['classifier.head.weight'] = model_dict['classifier.head.weight'][:num_class]
    model_dict['classifier.head.bias'] = model_dict['classifier.head.bias'][:num_class]
    return checkpoint




def revise_weights_6d(checkpoint, num_class=4):
    import torch  # Ensure torch is imported
    model_dict = checkpoint['model']
    device = 'cpu'
    num = 2
    k_multiply = [1, 2, 4, 8]
    layers_blocks = [2, 2, 8, 2]  # Number of blocks for each layer
    param_keys = [
        'op.x_proj_weight',
        'op.dt_projs_weight',
        'op.dt_projs_bias',
        'op.A_logs',
        'op.Ds',
    ]
    for i, num_blocks in enumerate(layers_blocks):
        for j in range(num_blocks):
            for key in param_keys:
                pkey = f'layers.{i}.blocks.{j}.{key}'
                params_exist = model_dict[pkey]
                if key in ['op.A_logs', 'op.Ds']:
                    add_len = 96 * num * k_multiply[i]
                    params_add = params_exist[:add_len]
                else:
                    params_add = params_exist[:num]
                params_new = torch.concat([params_exist, params_add], dim=0)
                model_dict[pkey] = params_new

    model_dict['classifier.head.weight'] = model_dict['classifier.head.weight'][:num_class]
    model_dict['classifier.head.bias'] = model_dict['classifier.head.bias'][:num_class]
    return checkpoint


def revise_weights_2d(checkpoint, num_class=4):
    model_dict = checkpoint['model']
    device = 'cpu'

    layers_blocks = [2, 2, 8, 2]  # Number of blocks for each layer
    param_keys = [
        'op.x_proj_weight',
        'op.dt_projs_weight',
        'op.dt_projs_bias',
        'op.A_logs',
        'op.Ds',
    ]

    for i, num_blocks in enumerate(layers_blocks):
        for j in range(num_blocks):
            for key in param_keys:
                pkey = f'layers.{i}.blocks.{j}.{key}'
                param = model_dict[pkey]
                num = param.size(0) // 2
                model_dict[pkey] = param[:num]

    # Revise classifier weights and bias
    model_dict['classifier.head.weight'] = model_dict['classifier.head.weight'][:num_class]
    model_dict['classifier.head.bias'] = model_dict['classifier.head.bias'][:num_class]

    return checkpoint

def revise_weights_1d(checkpoint, num_class=4):
    model_dict = checkpoint['model']
    device = 'cpu'

    layers_blocks = [2, 2, 8, 2]  # Number of blocks for each layer
    param_keys = [
        'op.x_proj_weight',
        'op.dt_projs_weight',
        'op.dt_projs_bias',
        'op.A_logs',
        'op.Ds',
    ]

    for i, num_blocks in enumerate(layers_blocks):
        for j in range(num_blocks):
            for key in param_keys:
                pkey = f'layers.{i}.blocks.{j}.{key}'
                param = model_dict[pkey]
                num = param.size(0) // 4
                model_dict[pkey] = param[:num]

    # Revise classifier weights and bias
    model_dict['classifier.head.weight'] = model_dict['classifier.head.weight'][:num_class]
    model_dict['classifier.head.bias'] = model_dict['classifier.head.bias'][:num_class]

    return checkpoint


def load_pretrained_revise(config, model, logger, model_ema: ModelEma = None, num_dir=6):
    ## by Shirley
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    if num_dir == 6:
        checkpoint = revise_weights_6d(checkpoint, num_class=config.MODEL.NUM_CLASSES)
    elif num_dir == 2:
        checkpoint = revise_weights_2d(checkpoint, num_class=config.MODEL.NUM_CLASSES)
    elif num_dir == 1:
        checkpoint = revise_weights_1d(checkpoint, num_class=config.MODEL.NUM_CLASSES)

    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded 'model' successfully from '{config.MODEL.PRETRAINED}'")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.PRETRAINED}! ")

    if model_ema is not None:
        if "model_ema" in checkpoint:
            logger.info(f"=> loading 'model_ema' separately...")
        key = "model_ema" if ("model_ema" in checkpoint) else "model"
        if key in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint[key], strict=False)
            logger.warning(msg)
            logger.info(f"=> loaded '{key}' successfully from '{config.MODEL.PRETRAINED}' for model_ema")
        else:
            logger.warning(f"No '{key}' found in {config.MODEL.PRETRAINED}! ")

    del checkpoint
    torch.cuda.empty_cache()


def load_pretrained_ema(config, model, logger, model_ema: ModelEma=None):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    
    if 'model' in checkpoint:
        checkpoint['model']['classifier.head.weight'] = torch.zeros([config.MODEL.NUM_CLASSES, 768]) #by Shirley: revise to actual number of class
        checkpoint['model']['classifier.head.bias'] = torch.zeros([config.MODEL.NUM_CLASSES]) #by Shirley: revise to actual number of class
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded 'model' successfully from '{config.MODEL.PRETRAINED}'")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.PRETRAINED}! ")

    if model_ema is not None:
        if "model_ema" in checkpoint:
            logger.info(f"=> loading 'model_ema' separately...")
        key = "model_ema" if ("model_ema" in checkpoint) else "model"
        if key in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint[key], strict=False)
            logger.warning(msg)
            logger.info(f"=> loaded '{key}' successfully from '{config.MODEL.PRETRAINED}' for model_ema")
        else:
            logger.warning(f"No '{key}' found in {config.MODEL.PRETRAINED}! ")

    del checkpoint
    torch.cuda.empty_cache()


def load_pretrained_coral_corn(config, model, logger, model_ema: ModelEma = None, model_type='corn'):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    #Change Backbone
    model_dict = checkpoint['model']
    device = 'cpu'
    num = 2
    k_multiply = [1, 2, 4, 8]
    layers_blocks = [2, 2, 8, 2]  # Number of blocks for each layer
    param_keys = [
        'op.x_proj_weight',
        'op.dt_projs_weight',
        'op.dt_projs_bias',
        'op.A_logs',
        'op.Ds',
    ]
    for i, num_blocks in enumerate(layers_blocks):
        for j in range(num_blocks):
            for key in param_keys:
                pkey = f'layers.{i}.blocks.{j}.{key}'
                params_exist = model_dict[pkey]
                if key in ['op.A_logs', 'op.Ds']:
                    add_len = 96 * num * k_multiply[i]
                    params_add = params_exist[:add_len]
                else:
                    params_add = params_exist[:num]
                params_new = torch.concat([params_exist, params_add], dim=0)
                model_dict[pkey] = params_new

    # Change Classifier Head
    if 'model' in checkpoint:
        if model_type == 'corn':
            checkpoint['model']['classifier.head.weight'] = torch.zeros(
                [config.MODEL.NUM_CLASSES-1, 768])  # by Shirley: revise to actual number of class
            checkpoint['model']['classifier.head.bias'] = torch.zeros(
                [config.MODEL.NUM_CLASSES-1])  # by Shirley: revise to actual number of class
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded 'model' successfully from '{config.MODEL.PRETRAINED}'")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.PRETRAINED}! ")

    if model_ema is not None:
        if "model_ema" in checkpoint:
            logger.info(f"=> loading 'model_ema' separately...")
        key = "model_ema" if ("model_ema" in checkpoint) else "model"
        if key in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint[key], strict=False)
            logger.warning(msg)
            logger.info(f"=> loaded '{key}' successfully from '{config.MODEL.PRETRAINED}' for model_ema")
        else:
            logger.warning(f"No '{key}' found in {config.MODEL.PRETRAINED}! ")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint_ema(config, epoch, model, val_max_accuracy, val_max_corrTst, tst_max_accuracy,  optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None, max_accuracy_ema=None, save_name=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'val_max_accuracy': val_max_accuracy,
                  'val_max_corrTst': val_max_corrTst,
                  'tst_max_accuracy': tst_max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    
    if model_ema is not None:
        save_state.update({'model_ema': model_ema.ema.state_dict(),
            'max_accuray_ema': max_accuracy_ema})

    if save_name == None:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    else:
        save_path = os.path.join(config.OUTPUT, f'{save_name}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

