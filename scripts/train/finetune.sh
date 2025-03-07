export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

deepspeed --include localhost:0,1 --master_port 29501 lvlm/train.py \
    --deepspeed scripts/train/utils/zero2.json \
    --data_path /hdd/shiym/datasets_processed/MedM/m3d_finetune.json \
    --conv_version llama3 \
    --image3d_path /hdd/shiym/datasets/medical-image-analysis/M3D/npys_256 \
    --training_recipe common \
    --tune_type_llm lora \
    --llm_lora_r 128 \
    --llm_lora_alpha 256 \
    --llm_lora_dropout 0.05 \
    --llm_lora_bias none \
    --tune_type_encoder_image3d frozen \
    --tune_type_connector_image3d full \
    --bf16 True \
    --gradient_checkpointing True \
    --output_dir work_dirs/MedM-VL-CT-3B-en-finetune \
    --resume_from_checkpoint work_dirs/MedM-VL-CT-3B-en \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --eval_strategy no \
    --save_strategy no \
    --report_to tensorboard \
    --logging_steps 1
