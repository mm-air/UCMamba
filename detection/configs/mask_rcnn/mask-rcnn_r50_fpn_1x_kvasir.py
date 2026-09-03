_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/kvasir_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


max_epochs = 100
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=20,                  # Save checkpoint every 20 epochs
        save_best='coco/segm_mAP',    # Save the model with highest Segmentation mAP
        rule='greater'                # The metric should be maximized
    )
)