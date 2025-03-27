export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="/hdd/shiym/work_dirs/MedM-VL/TinyLLaVA-phi2-siglip-finetune"

CUDA_VISIBLE_DEVICES=1 python lvlm/inference.py \
    --model_dtype float16 \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/mmbench_cn.json \
    --conv_version phi \
    --image_path /hdd/shiym/datasets/0_public/LLaVA/eval/mmbench/images \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0

python scripts/eval/llava_bench/eval_mmbench.py \
    --predict_path $MODEL_PATH/eval/mmbench_cn.json \
    --result_path $MODEL_PATH/eval/mmbench_cn_result.xlsx \
    --answer_path /hdd/shiym/datasets/0_public/LLaVA/eval/mmbench/mmbench_dev_cn_20231003.tsv
