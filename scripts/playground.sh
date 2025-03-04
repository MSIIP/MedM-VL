export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="work_dirs/MedM-VL-2D-3B-en"

CUDA_VISIBLE_DEVICES=0 python lvlm/playground.py \
    --model_dtype bfloat16 \
    --conv_version llama3 \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_length 2048 \
    --num_beams 1 \
    --temperature 0
