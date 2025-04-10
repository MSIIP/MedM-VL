export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="work_dirs/MedM-VL-CT-3B-en"

CUDA_VISIBLE_DEVICES=0 python lvlm/inference.py \
    --model_dtype bfloat16 \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/m3d/m3d_cap.json \
    --conv_version qwen2 \
    --image3d_path /hdd/shiym/datasets/medical-image-analysis/M3D/npys_256 \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0

CUDA_VISIBLE_DEVICES=0 python scripts/eval/m3d_bench/eval_m3d_cap.py \
    --answer_path /hdd/shiym/datasets_processed/MedM-VL/m3d/m3d_cap.json \
    --predict_path $MODEL_PATH/eval/m3d_cap.json \
    --result_path $MODEL_PATH/eval/m3d_cap_result.json
