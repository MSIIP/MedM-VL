export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="work_dirs/MedM-VL-2D-3B-en"

CUDA_VISIBLE_DEVICES=0 python lvlm/inference.py \
    --model_dtype bfloat16 \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/unimed/medmnist_derma.json \
    --conv_version qwen2 \
    --image_path / \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0
