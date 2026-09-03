# Training script example: single-node, single-GPU distributed launch
# Replace the paths below with your own dataset / pretrained checkpoint / output directory
DATA_PATH=/path/to/dataset                                    # Path to the dataset
PRETRAINED_CKPT=/path/to/vssm1_tiny_0230s_ckpt_epoch_264.pth  # Path to the pretrained checkpoint
OUTPUT_DIR=/path/to/output                                    # Directory to save logs/checkpoints
MODEL_NAME=model_name                                         # Name of the saved model


python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --master_addr="127.0.0.1" \
    --master_port=29501 \
    main.py \
    --cfg ./vmambav2v_tiny_224.yaml  \
    --data-path $DATA_PATH \
    --pretrained $PRETRAINED_CKPT \
    --model_ema False \
    --output $OUTPUT_DIR \
    --epochs 2 \
    --batch-size 32 \
    --lr 5e-3 \
    --tag $MODEL_NAME \
    --use_ctr True \
    --loss_ctr SeqCon \
    --weight_ctr 0.05 \
    --temper 0.2 \
    --accumulation-steps  1 \
    --ssm_forwardtype v05hvs_noz



