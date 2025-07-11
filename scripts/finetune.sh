#!/bin/bash

# Set environment variables
export FLASH_ATTENTION_FORCE_DISABLED=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/kaggle/working/Qwen2-VL-Finetune/"

# Define variables
MODEL_OUTPUT="output_model"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH="/kaggle/working/fine_tune/train"
IMAGE_FOLDER="vital_vision/images"
EVAL_DATA_PATH="/kaggle/working/fine_tune/validation"
INFERENCE_IMAGE_PATH="vital_vision/images/1/4.jpg"

# Make sure output directory exists
mkdir -p "$MODEL_OUTPUT"

# Clear GPU cache using Python (optional)
python3 -c "import torch; torch.cuda.empty_cache()"

cd /kaggle/working/Qwen2-VL-Finetune/
ls -F
cd vital_vision/
ls -F
cd images/
ls -F
cd ..
cd ..
ls -F

# Run training
accelerate launch src/train/train_sft.py \
    --deepspeed scripts/zero2_offload.json \
    --use_liger False \
    --data_path $DATA_PATH \
    --model_id $MODEL_NAME \
    --image_folder $IMAGE_FOLDER \
    --page_number 1 \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 False \
    --fp16 True \
    --lora_enable True \
    --disable_flash_attn2 True \
    --output_dir $MODEL_OUTPUT \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --save_strategy "epoch" \
    --save_steps 10 \
    --save_total_limit 10 \
    --dataloader_num_workers 0 \
    --per_device_train_batch_size 1 \
    --fp16_full_eval False \
    --image_min_pixels 1372 \
    --image_max_pixels 1372 \
    --bits 4 \
    --quant_type nf4 \
    --optim adamw_torch_fused \
    --optim_target_modules "q_proj,v_proj,o_proj" \
    --lora_namespan_exclude "['embed_tokens,lm_head']" \
    --num_train_epochs 1 \
    --do_eval True \
    --eval_strategy "epoch" \
    --eval_data_path $EVAL_DATA_PATH \
    --inference_image_path $INFERENCE_IMAGE_PATH
