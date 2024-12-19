export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="/hdd/shiym/work_dirs/MedM-VL/TinyLLaVA-phi2-siglip-finetune"

CUDA_VISIBLE_DEVICES=6 python lvlm/inference.py \
    --model_dtype float16 \
    --data_path /hdd/shiym/datasets_processed/MedM/pope/pope.json \
    --conv_version phi \
    --image_path /hdd/shiym/datasets/0_public/COCO/val2014 \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_length 3072 \
    --num_beams 1 \
    --temperature 0

python scripts/eval/pope/eval_pope.py \
    --question_path /hdd/shiym/datasets_processed/MedM/pope/pope.json \
    --predict_path $MODEL_PATH/eval/pope.json \
    --answer_dir /hdd/shiym/datasets/0_public/LLaVA/eval/pope/coco
