import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import transformers
from transformers import set_seed as hf_set_seed


@dataclass
class ModelArguments:
    cache_dir_hf: Optional[str] = field(default=None)

    # text
    llm_type: Optional[str] = field(default=None)
    llm_name_or_path: Optional[str] = field(default=None)
    llm_max_length: Optional[int] = field(default=None)
    llm_padding_side: Optional[str] = field(default=None)
    llm_attn_implementation: Optional[str] = field(default=None)
    tokenizer_use_fast: Optional[bool] = field(default=False)  # 虽然TokenizerFast快很多，但可能导致decode后与原文不一致

    # image
    encoder_image_type: Optional[str] = field(default=None)
    encoder_image_name_or_path: Optional[str] = field(default=None)
    encoder_image_select_layer: Optional[int] = field(default=None)
    encoder_image_select_feature: Optional[str] = field(default=None)
    connector_image_type: Optional[str] = field(default=None)
    connector_image_name: Optional[str] = field(default=None)
    connector_image_path: Optional[str] = field(default=None)

    # image3d
    encoder_image3d_type: Optional[str] = field(default=None)
    encoder_image3d_name_or_path: Optional[str] = field(default=None)
    encoder_image3d_select_layer: Optional[int] = field(default=None)
    encoder_image3d_select_feature: Optional[str] = field(default=None)
    connector_image3d_type: Optional[str] = field(default=None)
    connector_image3d_name: Optional[str] = field(default=None)
    connector_image3d_path: Optional[str] = field(default=None)

    # 设置print格式，每个属性打印一行，开头打印类名
    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            + ",\n".join(f"{k}={v}" for k, v in self.__dict__.items())
            + ",\n)"
        )


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    conv_version: str = field(default="pretrain")

    image_path: Optional[str] = field(default=None)
    image3d_path: Optional[str] = field(default=None)

    # 设置print格式，每个属性打印一行，开头打印类名
    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            + ",\n".join(f"{k}={v}" for k, v in self.__dict__.items())
            + ",\n)"
        )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_recipe: str = field(default="common")

    tune_type_llm: str = field(default="frozen")
    llm_lora_r: Optional[int] = field(default=None)
    llm_lora_alpha: Optional[int] = field(default=None)
    llm_lora_dropout: Optional[float] = field(default=None)
    llm_lora_bias: Optional[str] = field(default=None)

    tune_type_encoder_image: str = field(default="frozen")
    encoder_image_lora_r: Optional[int] = field(default=None)
    encoder_image_lora_alpha: Optional[int] = field(default=None)
    encoder_image_lora_dropout: Optional[float] = field(default=None)
    encoder_image_lora_bias: Optional[str] = field(default=None)
    tune_type_connector_image: str = field(default="frozen")

    tune_type_encoder_image3d: str = field(default="frozen")
    encoder_image3d_lora_r: Optional[int] = field(default=None)
    encoder_image3d_lora_alpha: Optional[int] = field(default=None)
    encoder_image3d_lora_dropout: Optional[float] = field(default=None)
    encoder_image3d_lora_bias: Optional[str] = field(default=None)
    tune_type_connector_image3d: str = field(default="frozen")


def set_seed(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Arguments:
        seed (int): The seed value to set.
    """

    # Set Python built-in random seed
    random.seed(seed)

    # Set numpy seed
    np.random.seed(seed)

    # Set PyTorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables the auto-tuner for convolution algorithms, enabling deterministic results

    # Set TensorFlow seed (if using TensorFlow)
    # try:
    #     tf.random.set_seed(seed)
    # except AttributeError:
    #     print("TensorFlow is not installed.")

    # Set random seed for OS-level operations (affects multiprocessing, etc.)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # If using other libraries like Hugging Face transformers, set seeds accordingly
    hf_set_seed(seed)

    print(f"Random seed set to {seed} for all libraries.")
