export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="work_dirs/MedM-VL-CT-Chest-3B-en"

CUDA_VISIBLE_DEVICES=1 python lvlm/inference.py \
    --model_dtype bfloat16 \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/ctrate/valid_long.json \
    --conv_version qwen2 \
    --image3d_path /hdd/common/datasets/medical-image-analysis/CT-RATE/dataset/preprocessed_32_256_256_raw/valid \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0 \
    --batch_size 16

CUDA_VISIBLE_DEVICES=1 python scripts/eval/ctrate/eval_ctrate_text.py \
    --answer_path /hdd/shiym/datasets_processed/MedM-VL/ctrate/valid_long.json \
    --predict_path $MODEL_PATH/eval/valid_long.json \
    --result_path $MODEL_PATH/eval/valid_long_result.json
