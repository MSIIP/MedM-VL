export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="work_dirs/MedM-VL-2D-3B-en"

CUDA_VISIBLE_DEVICES=0 python lvlm/inference.py \
    --resume_from_checkpoint $MODEL_PATH \
    --model_dtype bfloat16 \
    --data_path examples/data/inference_2d.json \
    --conv_version qwen2 \
    --image_dir / \
    --output_path $MODEL_PATH/eval/output.json \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0
