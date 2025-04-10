export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

deepspeed --include localhost:2,3 --master_port 29502 lvlm/train.py \
    --deepspeed scripts/train/utils/zero3.json \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/ctrate/train_vqa.json \
    --conv_version qwen2 \
    --image3d_path /hdd/common/datasets/medical-image-analysis/CT-RATE/dataset/preprocessed_32_256_256_raw/train \
    --training_recipe common \
    --tune_type_llm full \
    --tune_type_encoder_image3d full \
    --tune_type_connector_image3d full \
    --bf16 True \
    --gradient_checkpointing True \
    --output_dir /hdd/shiym/work_dirs/MedM-VL/MedM-VL-CT-Chest-3B-en-finetune-m3dclip \
    --resume_from_checkpoint /hdd/shiym/work_dirs/MedM-VL/MedM-VL-CT-Chest-3B-en-pretrain-m3dclip \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --eval_strategy no \
    --save_strategy no \
    --report_to tensorboard \
    --logging_steps 1
