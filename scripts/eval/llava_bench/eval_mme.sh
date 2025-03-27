export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="/hdd/shiym/work_dirs/MedM-VL/TinyLLaVA-phi2-siglip-finetune"
MODEL_NAME="TinyLLaVA-phi2-siglip-finetune"

CUDA_VISIBLE_DEVICES=3 python lvlm/inference.py \
    --model_dtype float16 \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/mme.json \
    --conv_version phi \
    --image_path /hdd/shiym/datasets/0_public/LLaVA/eval/MME/MME_Benchmark_release_version \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0

python scripts/eval/llava_bench/eval_mme.py \
    --question_path /hdd/shiym/datasets_processed/MedM-VL/llava/mme.json \
    --predict_path $MODEL_PATH/eval/mme.json \
    --answer_dir /hdd/shiym/datasets/0_public/LLaVA/eval/MME/MME_Benchmark_release_version \
    --output_dir /hdd/shiym/datasets/0_public/LLaVA/eval/MME/eval_tool/answers/$MODEL_NAME \

cd /hdd/shiym/datasets/0_public/LLaVA/eval/MME/eval_tool
python calculation.py --results_dir answers/$MODEL_NAME
