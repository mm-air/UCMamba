# Inherit from a standard config (Mask R-CNN ResNet50)
_base_ = 'mmdet::mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

# 1. Dataset Settings
dataset_type = 'CocoDataset'
data_root = '/home/shirley/Desktop/Projects/Dataset/kvasir_seg_raw/Kvasir-SEG/'
classes = ('polyp',)

# 2. Model Head Modifications (Number of classes)
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    )
)

# 3. Data Loaders
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(classes=classes)
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(classes=classes)
    )
)

test_dataloader = val_dataloader

# 4. Evaluator (Use COCO metric but for our class)
val_evaluator = dict(
    ann_file=data_root + 'val.json',
    metric=['bbox', 'segm']
)
test_evaluator = val_evaluator

# 5. Runtime settings
train_cfg = dict(max_epochs=12) # Train for 12 epochs
# Adjust learning rate for small batch size (standard is 0.02 for batch 16)
# For batch size 2, use 0.02 / 8 = 0.0025
optim_wrapper = dict(optimizer=dict(lr=0.0025))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
