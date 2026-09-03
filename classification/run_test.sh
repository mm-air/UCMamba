# Evaluation script example: run inference/metrics with a trained checkpoint
# Replace the paths below with your own directories
DATA_PATH=/path/to/dataset              # Path to the dataset
PRETRAINED_CKPT=/path/to/vssm1_tiny_0230s_ckpt_epoch_264.pth  # Path to the backbone pretrained weights
MODEL_CKPT=/path/to/val_best_model.pth    # Path to the trained checkpoint to be evaluated
SAVE_PATH=/path/to/eval_output            # Directory to save evaluation results
TAG_NAME=model_name  # test results folder tag, usually match how the model was trained/named

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --master_addr="127.0.0.1" \
    --master_port=25678 \
    eval_all_metrics.py \
    --cfg ./vmambav2v_tiny_224.yaml \
    --data-path $DATA_PATH \
    --pretrained $PRETRAINED_CKPT \
    --model_ema False \
    --output $SAVE_PATH \
    --batch-size 128 \
    --tag $TAG_NAME \
    --eval_model $MODEL_CKPT \
    --use_ctr True \
    --loss_ctr SeqCon \
    --weight_ctr 0.05 \
    --temper 0.2 \
    --accumulation-steps 1 \
    --ssm_forwardtype v05hvs_noz         # Forward computation type of the state-space model (SSM)

