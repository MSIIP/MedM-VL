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
    # text
    lvlm_llm_type: Optional[str] = field(default=None)
    lvlm_llm_name_or_path: Optional[str] = field(default=None)
    lvlm_llm_max_length: Optional[int] = field(default=None)
    lvlm_llm_padding_side: Optional[str] = field(default=None)
    lvlm_llm_attn_implementation: Optional[str] = field(default=None)

    # image
    lvlm_encoder_image_type: Optional[str] = field(default=None)
    lvlm_encoder_image_name_or_path: Optional[str] = field(default=None)
    lvlm_encoder_image_select_layer: Optional[int] = field(default=None)
    lvlm_encoder_image_select_feature: Optional[str] = field(default=None)
    lvlm_connector_image_type: Optional[str] = field(default=None)
    lvlm_connector_image_name: Optional[str] = field(default=None)
    lvlm_connector_image_path: Optional[str] = field(default=None)

    # image3d
    lvlm_encoder_image3d_type: Optional[str] = field(default=None)
    lvlm_encoder_image3d_name_or_path: Optional[str] = field(default=None)
    lvlm_encoder_image3d_select_layer: Optional[int] = field(default=None)
    lvlm_encoder_image3d_select_feature: Optional[str] = field(default=None)
    lvlm_connector_image3d_type: Optional[str] = field(default=None)
    lvlm_connector_image3d_name: Optional[str] = field(default=None)
    lvlm_connector_image3d_path: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    data_path: str = field(default=None)
    conv_version: str = field(default="pretrain")
    image_dir: Optional[str] = field(default=None)
    image3d_dir: Optional[str] = field(default=None)

    training_recipe: str = field(default="common")

    tune_type_llm: str = field(default="frozen")
    llm_lora_r: Optional[int] = field(default=None)
    llm_lora_alpha: Optional[int] = field(default=None)
    llm_lora_dropout: Optional[float] = field(default=None)
    llm_lora_bias: Optional[str] = field(default=None)

    tune_type_encoder_image: str = field(default="frozen")
    tune_type_connector_image: str = field(default="frozen")

    tune_type_encoder_image3d: str = field(default="frozen")
    tune_type_connector_image3d: str = field(default="frozen")

    ratio_json: str = field(default=None, metadata={"help": "extra args"})
    multiloader: bool = field(default=False, metadata={"help": "use multiloader"})


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
