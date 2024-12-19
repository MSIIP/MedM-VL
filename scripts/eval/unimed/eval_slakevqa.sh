export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="/hdd/shiym/work_dirs/MedM-VL/MedM-VL-2D-3B-en"

CUDA_VISIBLE_DEVICES=7 python lvlm/inference.py \
    --model_dtype bfloat16 \
    --data_path /hdd/shiym/datasets_processed/MedM/unimed/slakevqa.json \
    --conv_version llama3 \
    --image_path / \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_length 2048 \
    --num_beams 1 \
    --temperature 0
