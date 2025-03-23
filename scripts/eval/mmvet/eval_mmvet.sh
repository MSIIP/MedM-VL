export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="/hdd/shiym/work_dirs/MedM-VL/TinyLLaVA-phi2-siglip-finetune"

CUDA_VISIBLE_DEVICES=5 python lvlm/inference.py \
    --model_dtype float16 \
    --data_path /hdd/shiym/datasets_processed/MedM/mmvet/mmvet.json \
    --conv_version phi \
    --image_path /hdd/shiym/datasets/0_public/LLaVA/eval/mm-vet/images \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0

python scripts/eval/mmvet/eval_mmvet.py \
    --predict_path $MODEL_PATH/eval/mmvet.json \
    --result_path $MODEL_PATH/eval/mmvet_result.json
