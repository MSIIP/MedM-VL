export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

deepspeed --include localhost:0,1,2,3 --master_port 29501 lvlm/train.py \
    --deepspeed scripts/train/utils/zero3.json \
    --cache_dir_hf /mnt/nfs_share/shiym/ckpts/cache_dir_hf \
    --llm_type llama3 \
    --llm_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --llm_max_length 2048 \
    --llm_padding_side right \
    --llm_attn_implementation flash_attention_2 \
    --tokenizer_use_fast False \
    --encoder_image_type siglip \
    --encoder_image_name_or_path google/siglip-so400m-patch14-384 \
    --encoder_image_select_layer -1 \
    --encoder_image_select_feature cls_patch \
    --connector_image_type mlp \
    --connector_image_name mlp2x_gelu \
    --data_path /hdd/shiym/datasets_processed/MedM/llava_pretrain.json \
    --conv_version pretrain \
    --image_path /hdd/shiym/datasets/0_public/LLaVA/LLaVA-Pretrain/images \
    --training_recipe common \
    --tune_type_llm frozen \
    --tune_type_encoder_image frozen \
    --tune_type_connector_image full \
    --bf16 True \
    --gradient_checkpointing True \
    --output_dir /hdd/shiym/work_dirs/MedM-VL/MedM-VL-2D-3B-en-pretrain \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --eval_strategy no \
    --save_strategy no \
    --report_to tensorboard \
    --logging_steps 1
