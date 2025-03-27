export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="/hdd/shiym/work_dirs/MedM-VL/TinyLLaVA-phi2-siglip-finetune"

CUDA_VISIBLE_DEVICES=4 python lvlm/inference.py \
    --model_dtype float16 \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/mmmu.json \
    --conv_version phi \
    --image_path /hdd/shiym/datasets/0_public/LLaVA/eval/MMMU/all_images \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0

python scripts/eval/llava_bench/eval_mmmu.py \
    --answer_path /home/shiym/datasets_processed/medmllm/mmmu/anns_raw.json \
    --predict_path $MODEL_PATH/eval/mmmu.json \
    --result_path $MODEL_PATH/eval/mmmu_result.json

python /hdd/shiym/datasets/0_public/LLaVA/eval/MMMU/eval/main_eval_only.py \
    --output_path $MODEL_PATH/eval/mmmu_result.json \
    --answer_path /hdd/shiym/datasets/0_public/LLaVA/eval/MMMU/eval/answer_dict_val.json
