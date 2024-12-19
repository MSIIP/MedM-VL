export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="/hdd/shiym/work_dirs/MedM-VL/TinyLLaVA-phi2-siglip-finetune"

CUDA_VISIBLE_DEVICES=7 python lvlm/inference.py \
    --model_dtype float16 \
    --data_path /hdd/shiym/datasets_processed/MedM/sqa/sqa.json \
    --conv_version phi \
    --image_path /hdd/shiym/datasets/0_public/LLaVA/eval/scienceqa/test \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_length 3072 \
    --num_beams 1 \
    --temperature 0

python scripts/eval/sqa/eval_sqa.py \
    --answer_path /hdd/shiym/datasets_processed/MedM/sqa/sqa.json \
    --predict_path $MODEL_PATH/eval/sqa.json \
    --dataset_split /hdd/shiym/datasets/0_public/LLaVA/eval/scienceqa/pid_splits.json
