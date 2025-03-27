import copy
import json
import os.path as osp
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from lvlm.dataset.template import TEMPlATE_FACTORY


class MultiModalDataset(Dataset):
    def __init__(self, model, data_arguments, mode):
        super(MultiModalDataset, self).__init__()
        self.data_arguments = data_arguments
        self.mode = mode
        with open(data_arguments.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

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
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_dict = self.template.encode(
            messages=copy.deepcopy(data_item["conversations"]),
            tokenizer=self.tokenizer,
            mode=self.mode,
        )
        # print("-" * 30 + "item" + "-" * 30)
        # print("before preprocess" + "-" * 29)
        # for k, v in data_item.items():
        #     if isinstance(v, torch.Tensor):
        #         v = v[:10]
        #     print(f"{k:16}", v)
        # print("after preprocess" + "-" * 30)
        # for k, v in data_dict.items():
        #     if isinstance(v, torch.Tensor):
        #         v = v[:10]
        #     print(f"{k:16}", v)

        if "image" in data_item:  # for multi image
            data_dict["image"] = []
            for filename in data_item["image"]:
                image_path = osp.join(self.data_arguments.image_path, filename)
                image = Image.open(image_path).convert("RGB")
                image = self.preprocessor_image(image, mode=self.mode)
                data_dict["image"].append(image)
        else:
            data_dict["image"] = None

        if "image3d" in data_item:
            data_dict["image3d"] = []
            for filename in data_item["image3d"]:
                image3d_path = osp.join(self.data_arguments.image3d_path, filename)
                image3d = np.load(image3d_path)
                image3d = self.preprocessor_image3d(image3d, mode=self.mode)
                data_dict["image3d"].append(image3d)
        else:
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

        if self.mode == "train":
            labels = [instance["labels"] for instance in instances]
            if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
                for label in labels:
                    label[label == self.tokenizer.eos_token_id] = -300

            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = labels[:, : self.tokenizer.model_max_length]

            if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
                for label in labels:
                    label[label == -300] = self.tokenizer.eos_token_id
            batch["labels"] = labels

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


def create_data_module(model, data_arguments, mode):
    train_dataset = MultiModalDataset(model=model, data_arguments=data_arguments, mode=mode)
    data_collator = DataCollatorForMultiModalDataset(tokenizer=model.tokenizer, mode=mode)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
