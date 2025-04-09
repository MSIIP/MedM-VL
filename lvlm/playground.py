import argparse
from tqdm import tqdm

import gradio as gr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lvlm.dataset.template import TEMPlATE_FACTORY
from lvlm.model.modeling_lvlm import LVLMForConditionalGeneration
from lvlm.utils.arguments import set_seed


class MultiModalDataset(Dataset):
    def __init__(self, model, data_arguments, mode, image, prompt):
        super(MultiModalDataset, self).__init__()
        self.data_arguments = data_arguments
        self.mode = mode
        self.image = image
        self.prompt = "<image>\n" + prompt
        # with open(data_arguments.data_path, "r") as f:
        #     self.data = json.load(f)

        self.tokenizer = model.tokenizer
        self.template = TEMPlATE_FACTORY[data_arguments.conv_version]()

        if model.encoder_image is not None:
            self.preprocessor_image = model.encoder_image.processor
        else:
            self.preprocessor_image = None

        if model.encoder_image3d is not None:
            self.preprocessor_image3d = model.encoder_image3d.processor
        else:
            self.preprocessor_image3d = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # data_item = self.data[idx]
        conversations = [
            {
                "from": "human",
                "value": self.prompt
            },
        ]
        data_dict = self.template.encode(
            messages=conversations,
            tokenizer=self.tokenizer,
            mode=self.mode,
        )
        image = self.preprocessor_image(self.image, mode=self.mode)
        data_dict["image"] = []
        data_dict["image"].append(image)
        data_dict["image3d"] = None
        return data_dict


class DataCollatorForMultiModalDataset:
    def __init__(self, tokenizer, mode):
        self.tokenizer = tokenizer
        self.mode = mode

    def __call__(self, instances):
        input_ids = [instance["input_ids"] for instance in instances]
        if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        image_list = []
        for instance in instances:
            if instance["image"] is not None:
                image_list.extend(instance["image"])  # for multi image
        image = torch.stack(image_list) if len(image_list) > 0 else None
        batch["image"] = image

        image3d_list = []
        for instance in instances:
            if instance["image3d"] is not None:
                image3d_list.extend(instance["image3d"])
        image3d = torch.stack(image3d_list) if len(image3d_list) > 0 else None
        batch["image3d"] = image3d

        return batch


def create_data_module(model, data_arguments, mode, image, prompt):
    train_dataset = MultiModalDataset(model=model, data_arguments=data_arguments, mode=mode, image=image, prompt=prompt)
    data_collator = DataCollatorForMultiModalDataset(tokenizer=model.tokenizer, mode=mode)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


def predict(prompt, image, temperature, max_token):
    img = image.convert("RGB")
    data_module = create_data_module(
        model=model,
        data_arguments=args,
        mode="eval",
        image=img,
        prompt=prompt,
    )
    data_loader = DataLoader(
        data_module["train_dataset"],
        batch_size=1,
        shuffle=False,
        collate_fn=data_module["data_collator"],
    )
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            for k, v in batch.items():
                if v is not None:
                    batch[k] = v.to(device)
                    if k == "image" or k == "image3d":
                        batch[k] = v.to(dtype=model_dtype, device=device)

            output_ids = model.generate(
                **batch,
                max_new_tokens=max_token,
                do_sample=True if temperature > 0 else False,
                num_beams=args.num_beams,
                temperature=temperature,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
            )

            outputs = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dtype', type=str)
    parser.add_argument('--conv_version', type=str)
    parser.add_argument('--resume_from_checkpoint', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--new_length', type=int)
    parser.add_argument('--num_beams', type=int)
    parser.add_argument('--temperature', type=int)
    args = parser.parse_args()
    set_seed(42)

    print("loading model")
    model_dtype = torch.float16 if args.model_dtype == "float16" else (torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LVLMForConditionalGeneration.from_pretrained(args.resume_from_checkpoint)
    model.to(dtype=model_dtype, device=device)
    model.eval()
    print("loading success")

    # Launch the demo
    with gr.Blocks() as demo:
        gr.Markdown("""
        <style>
            h1 {
                font-size: 28px !important;
            }
            label {
                font-size: 18px !important;
            }
            .textbox, .button {
                font-size: 16px !important;
            }
        </style>
        """)
        gr.Markdown("# 🚀 MedM-VL 交互框架")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="输入文本提示", placeholder="请输入您的提示...")
                image_input = gr.Image(type="pil", label="上传图片")
                
                temperature = gr.Slider(
                    minimum=0, maximum=1, step=0.1, value=0,
                    label="Temperature", interactive=True
                )
                max_tokens = gr.Slider(
                    minimum=0, maximum=1024, step=1, value=256,
                    label="Max output tokens", interactive=True
                )
                with gr.Row():
                    submit_btn = gr.Button("提交", variant="primary")
                    clear_btn = gr.Button("全部清除", variant="secondary")
            
            with gr.Column():
                output_text = gr.Textbox(label="模型输出", lines=10)

        submit_btn.click(
            fn=predict,
            inputs=[text_input, image_input, temperature, max_tokens], 
            outputs=output_text
        )

        clear_btn.click(
            lambda: [None, None, "", 0.0, 256], 
            outputs=[text_input, image_input, output_text, temperature, max_tokens] 
        )
    demo.launch()
