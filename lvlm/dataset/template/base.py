from dataclasses import dataclass
import copy
import re

import torch

from lvlm.dataset.template.formatter import Formatter
from lvlm.utils.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_ID,
    DEFAULT_IMAGE3D_TOKEN,
    IMAGE3D_TOKEN_ID,
)

special_token_dict = {
    DEFAULT_IMAGE_TOKEN: IMAGE_TOKEN_ID,
    DEFAULT_IMAGE3D_TOKEN: IMAGE3D_TOKEN_ID,
}


@dataclass
class Template:
    system: "Formatter"
    format_user: "Formatter"
    format_assistant: "Formatter"
    separator: "Formatter"

    def encode(self, messages, tokenizer, mode="train"):
        # 将messages中的用户问题和助手回答分别提取出来组成两个list
        question_list, answer_list = self.get_list_from_message(messages)
        # 将问题和回答组合成一个prompt，其中加入系统提示、"USER: "、"ASSISTANT: "等形式化内容
        prompt = self.prompt(question_list, answer_list)
        # 将prompt转换为input_ids，其中bos_token，DEFAULT_IMAGE_TOKEN等special_token，且bos_token最多只有最前面有一个
        input_ids = self.tokenizer_special_token(prompt, tokenizer, return_tensors="pt")
        if mode == "train":
            labels = self.make_labels(input_ids, prompt, tokenizer)
            return dict(input_ids=input_ids, prompt=prompt, labels=labels)
        else:
            return dict(input_ids=input_ids, prompt=prompt)

    def get_list_from_message(self, messages):
        question_list = []
        answer_list = []
        first_is_not_question = 0

        for i, message in enumerate(messages):
            if i == 0 and message["from"] != "human":
                first_is_not_question = 1
                continue

            if i % 2 == first_is_not_question:
                question_list.append(message["value"])
            else:
                answer_list.append(message["value"])

        return question_list, answer_list

    def prompt(self, question_list, answer_list):
        # for generation process
        if len(question_list) == len(answer_list) + 1:
            answer_list.append("")

        msg = ""
        for i, (question, answer) in enumerate(zip(question_list, answer_list)):
            if i == 0:
                msg += self.system.apply()

            # if DEFAULT_IMAGE_TOKEN in question:
            #     question = question.replace(DEFAULT_IMAGE_TOKEN, "").strip()
            #     question = self.format_image_token.apply(content=question).strip()

            # for interleaved format
            question = question.strip()
            answer = answer.strip()

            msg += self.format_user.apply(content=question)
            if len(answer) != 0:
                msg += self.format_assistant.apply(content=answer)
            else:
                msg += "ASSISTANT:"  # for generation process

        return msg

    def tokenizer_special_token(self, prompt, tokenizer, return_tensors=None):
        # 检查是否有bos_token
        if tokenizer("test string").input_ids[0] == tokenizer.bos_token_id:
            offset = 1
        else:
            offset = 0

        # 使用正则表达式根据special_token_dict对字符串进行划分
        pattern = "|".join(re.escape(token) for token in special_token_dict)
        chunks = re.split(f"({pattern})", prompt)

        input_ids = []
        for chunk in chunks:
            if chunk in special_token_dict:
                input_ids.append(special_token_dict[chunk])
            elif len(chunk) > 0:
                # 首轮对话的第一个token可能需要手动添加bos_token
                if len(input_ids) == 0 and offset == 1:
                    input_ids.append(tokenizer.bos_token_id)
                input_ids.extend(tokenizer(chunk).input_ids[offset:])

        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)

        # for pre-training process
        return input_ids

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        sep, eos_token = self.separator.apply()

        # 计算labels中非padding token的总数
        total_len = int(labels.ne(tokenizer.pad_token_id).sum())
        # 如果pad_token_id等于eos_token_id，则需要额外加上prompt中eos_token的数量
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            total_len += prompt.count(eos_token)

        rounds = prompt.split(eos_token)
        eos_token_length = len(tokenizer.encode(eos_token))
        labels, cur_len = self._make_masks(labels, tokenizer, sep, eos_token_length, rounds)

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print("number of rounds: ", len(rounds) - 1)
                print("rounds: ", rounds[:-1])
                print("prompt: ", prompt)
                print(labels)
                print(input_ids)
                labels[:] = IGNORE_INDEX

        return labels

    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_pos = 0

        for idx, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            # rou里面的eos_token已经在make_labels里被删去了，所以这里需要加上eos_token_length(=1)
            rou_len = len(self.tokenizer_special_token(rou, tokenizer)) + 1
            instruction_len = len(self.tokenizer_special_token(parts[0], tokenizer))
            # 如果tokenizer会自动加入bos_token，则对于非首轮对话需要减去1
            if eos_token_length > 1 and idx != 0:
                rou_len = rou_len - 1
                instruction_len = instruction_len - 1

            labels[cur_pos : cur_pos + instruction_len] = IGNORE_INDEX
            cur_pos += rou_len

        labels[cur_pos:] = IGNORE_INDEX
        return labels, cur_pos
