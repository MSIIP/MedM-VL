export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="work_dirs/MedM-VL-CT-3B-en"

CUDA_VISIBLE_DEVICES=0 python lvlm/inference.py \
    --model_dtype bfloat16 \
    --data_path /hdd/shiym/datasets_processed/MedM/m3d_bench/m3d_cap.json \
    --conv_version llama3 \
    --image3d_path /hdd/shiym/datasets/medical-image-analysis/M3D/npys_256 \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_length 512 \
    --num_beams 1 \
    --temperature 0
