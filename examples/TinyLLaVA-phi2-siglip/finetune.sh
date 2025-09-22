export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

deepspeed --include localhost:0,1,2,3 --master_port 29501 lvlm/train.py \
    --deepspeed examples/zero3.json \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/finetune_llava.json \
    --conv_version phi \
    --image_path /hdd/shiym/datasets/0_public/LLaVA/visual-instruction-tuning-images \
    --training_recipe common \
    --tune_type_llm full \
    --tune_type_encoder_image frozen \
    --tune_type_connector_image full \
    --fp16 True \
    --gradient_checkpointing True \
    --output_dir /hdd/shiym/work_dirs/MedM-VL/TinyLLaVA-phi2-siglip-finetune \
    --resume_from_checkpoint /hdd/shiym/work_dirs/MedM-VL/TinyLLaVA-phi2-siglip-pretrain \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --eval_strategy no \
    --save_strategy no \
    --report_to tensorboard \
    --logging_steps 1
