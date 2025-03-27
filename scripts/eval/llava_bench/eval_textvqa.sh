export PYTHONPATH=$PYTHONPATH:/home/shiym/projects/MedM-VL

MODEL_PATH="/hdd/shiym/work_dirs/MedM-VL/TinyLLaVA-phi2-siglip-finetune"

CUDA_VISIBLE_DEVICES=7 python lvlm/inference.py \
    --model_dtype float16 \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/textvqa.json \
    --conv_version phi \
    --image_path /hdd/shiym/datasets/0_public/LLaVA/eval/textvqa/train_images \
    --resume_from_checkpoint $MODEL_PATH \
    --output_dir $MODEL_PATH/eval \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0

python scripts/eval/llava_bench/eval_textvqa.py \
    --question_path /hdd/shiym/datasets_processed/MedM-VL/llava/textvqa.json \
    --predict_path $MODEL_PATH/eval/textvqa.json \
    --answer_path /hdd/shiym/datasets/0_public/LLaVA/eval/textvqa/TextVQA_0.5.1_val.json
