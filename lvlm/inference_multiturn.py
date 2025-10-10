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

    with open(args.data_path, "r") as f:
        data_list = json.load(f)

    print("*" * 30 + "Stage 2" + "*" * 30)
    print("Inference...")
    with torch.no_grad():
        outputs_list = []

        for data in tqdm(data_list):
            num_turn = (len(data["conversations"]) + 1) // 2
            data_item = data.copy()
            data_item["conversations"] = []

            for idx in range(num_turn):
                data_item["conversations"].append(data["conversations"][2 * idx])  # Get the human input
                print(f"[{data_item['conversations'][-1]['from']}]:\n{data_item['conversations'][-1]['value']}\n")

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
                    batch_size=1,
                    shuffle=False,
                    collate_fn=data_module["data_collator"],
                )

                for batch in data_loader:
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
                    data_item["conversations"].append(
                        {
                            "from": "gpt",
                            "value": outputs[0].strip()
                        }
                    )
                    print(f"[{data_item['conversations'][-1]['from']}]:\n{data_item['conversations'][-1]['value']}\n")

            outputs_list.append(data_item)

    print("*" * 30 + "Stage 3" + "*" * 30)
    print("Save outputs...")
    os.makedirs(osp.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(outputs_list, f, indent=4, ensure_ascii=False)


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
