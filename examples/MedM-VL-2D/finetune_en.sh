export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

deepspeed --include localhost:0,1 --master_port 29501 lvlm/train.py \
    --deepspeed examples/zero2.json \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/unimed/train_unimed.json \
    --conv_version qwen2 \
    --image_dir / \
    --training_recipe common \
    --tune_type_llm full \
    --tune_type_encoder_image full \
    --tune_type_connector_image full \
    --resume_from_checkpoint /hdd/shiym/work_dirs/MedM-VL/MedM-VL-2D-3B-en-pretrain \
    --output_dir /hdd/shiym/work_dirs/MedM-VL/MedM-VL-2D-3B-en-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --bf16 True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --gradient_checkpointing True \
    --logging_strategy steps \
    --logging_steps 1 \
    --report_to tensorboard \
    --eval_strategy no \
    --save_strategy no
