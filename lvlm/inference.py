import argparse
import json
import os
import os.path as osp
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from lvlm.utils.arguments import set_seed
from lvlm.dataset.dataset import create_data_module
from lvlm.model.modeling_lvlm import LVLMForConditionalGeneration


def inference(args):
    model_dtype = torch.float16 if args.model_dtype == "float16" else (torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32)
    model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("*" * 30 + "Stage 1" + "*" * 30)
    print("Load model...")
    model = LVLMForConditionalGeneration.from_pretrained(args.resume_from_checkpoint)
    model.to(dtype=model_dtype, device=model_device)
    model.eval()

    if args.batch_size > 1:
        model.config.lvlm_llm_padding_side = "left"

    print("*" * 30 + "Stage 2" + "*" * 30)
    print("Create data_module...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        answer = ""
        if d["conversations"][-1]["from"] == "gpt":
            answer = d["conversations"][-1]["value"]
            d["conversations"] = d["conversations"][:-1]
        if "answer" not in d:
            d["answer"] = answer

    data_module = create_data_module(
        data=data,
        conv_version=args.conv_version,
        image_dir=args.image_dir,
        image3d_dir=args.image3d_dir,
        model=model,
        mode="eval",
    )
    data_loader = DataLoader(
        data_module["train_dataset"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_module["data_collator"],
    )

    print("*" * 30 + "Stage 4" + "*" * 30)
    print("Inference...")
    with torch.no_grad():
        outputs_list = []
        for batch in tqdm(data_loader, total=len(data_loader)):
            for k, v in batch.items():
                if v is not None:
                    batch[k] = v.to(model_device)
                    if k == "image" or k == "image3d":
                        batch[k] = v.to(dtype=model_dtype, device=model_device)

            output_ids = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=True if args.temperature > 0 else False,
                num_beams=args.num_beams,
                temperature=args.temperature,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
            )
            # output_ids = [output_ids_cur[len(input_ids):] for input_ids, output_ids_cur in zip(batch["input_ids"], output_ids)]  # for only text

            outputs = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs_list.extend(outputs)
            for output in outputs:
                print(output)

    for d, output in zip(data, outputs_list):
        d["conversations"].append({"from": "gpt", "value": output})

    print("*" * 30 + "Stage 5" + "*" * 30)
    print("Save outputs...")
    os.makedirs(osp.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--model_dtype", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--conv_version", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--image3d_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    set_seed(42)

    inference(args)
