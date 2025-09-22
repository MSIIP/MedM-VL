import argparse
import json
import os
import os.path as osp
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from lvlm.utils.arguments import set_seed
from lvlm.dataset.dataset import MultiModalDataset, DataCollatorForMultiModalDataset
from lvlm.model.modeling_lvlm import LVLMForConditionalGeneration


def inference(args):
    print("*" * 30 + "Stage 1" + "*" * 30)
    print("Load model...")
    model_dtype = torch.float16 if args.model_dtype == "float16" else (torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LVLMForConditionalGeneration.from_pretrained(args.resume_from_checkpoint)
    model.to(dtype=model_dtype, device=device)
    model.eval()

    if args.batch_size > 1:
        model.config.llm_padding_side = "left"

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

                train_dataset = MultiModalDataset(
                    model=model,
                    data=[data_item],
                    data_arguments=args,
                    mode="eval",
                )
                data_collator = DataCollatorForMultiModalDataset(tokenizer=model.tokenizer, mode="eval")
                data_loader = DataLoader(
                    train_dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=data_collator,
                )

                for batch in data_loader:
                    for k, v in batch.items():
                        if v is not None:
                            batch[k] = v.to(device)
                            if k == "image" or k == "image3d":
                                batch[k] = v.to(dtype=model_dtype, device=device)

                    output_ids = model.generate(
                        **batch,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True if args.temperature > 0 else False,
                        num_beams=args.num_beams,
                        temperature=args.temperature,
                        pad_token_id=model.tokenizer.pad_token_id,
                        eos_token_id=model.tokenizer.eos_token_id,
                    )

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
    os.makedirs(args.output_dir, exist_ok=True)
    with open(osp.join(args.output_dir, osp.basename(args.data_path)), "w") as f:
        json.dump(outputs_list, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dtype", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--conv_version", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--image3d_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    set_seed(42)

    inference(args)
