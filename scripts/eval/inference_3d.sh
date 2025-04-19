export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="work_dirs/MedM-VL-CT-Chest-3B-en"

CUDA_VISIBLE_DEVICES=0 python lvlm/inference.py \
    --model_dtype bfloat16 \
    --data_path docs/example_3d_inference.json \
    --conv_version qwen2 \
    --image3d_path /hdd/common/datasets/medical-image-analysis/CT-RATE/dataset/preprocessed_raw/valid \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0
