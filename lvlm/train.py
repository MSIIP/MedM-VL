import json
import random
import os
import os.path as osp
from types import SimpleNamespace

import transformers

from lvlm.utils.arguments import ModelArguments, DataArguments, TrainingArguments, set_seed
from lvlm.dataset.dataset import create_data_module, create_multi_data_module
from lvlm.model.configuration_lvlm import LVLMConfig
from lvlm.model.modeling_lvlm import LVLMForConditionalGeneration
from lvlm.utils.training_recipe import RECIPE_FACTORY
from lvlm.utils.trainer_lvlm import LVLMTrainer, LVLMMULTITrainer


def save_args(model_arguments, data_arguments, training_arguments, model_config):
    output_dir = osp.join(training_arguments.output_dir, "args")
    os.makedirs(output_dir, exist_ok=True)

    # update model_arguments through model_config
    model_arguments.cache_dir_hf = model_config.cache_dir_hf

    model_arguments.llm_type = model_config.llm_type
    model_arguments.llm_name_or_path = model_config.llm_name_or_path
    model_arguments.llm_max_length = model_config.llm_max_length
    model_arguments.llm_padding_side = model_config.llm_padding_side
    model_arguments.llm_attn_implementation = model_config.llm_attn_implementation
    model_arguments.tokenizer_use_fast = model_config.tokenizer_use_fast

    model_arguments.encoder_image_type = model_config.encoder_image_type
    model_arguments.encoder_image_name_or_path = model_config.encoder_image_name_or_path
    model_arguments.encoder_image_select_layer = model_config.encoder_image_select_layer
    model_arguments.encoder_image_select_feature = model_config.encoder_image_select_feature
    model_arguments.connector_image_type = model_config.connector_image_type
    model_arguments.connector_image_name = model_config.connector_image_name
    model_arguments.connector_image_path = model_config.connector_image_path

    model_arguments.encoder_image3d_type = model_config.encoder_image3d_type
    model_arguments.encoder_image3d_name_or_path = model_config.encoder_image3d_name_or_path
    model_arguments.encoder_image3d_select_layer = model_config.encoder_image3d_select_layer
    model_arguments.encoder_image3d_select_feature = model_config.encoder_image3d_select_feature
    model_arguments.connector_image3d_type = model_config.connector_image3d_type
    model_arguments.connector_image3d_name = model_config.connector_image3d_name
    model_arguments.connector_image3d_path = model_config.connector_image3d_path

    # save model arguments
    with open(osp.join(output_dir, "model_arguments.txt"), "w") as f:
        f.write(str(model_arguments))
    # save data arguments
    with open(osp.join(output_dir, "data_arguments.txt"), "w") as f:
        f.write(str(data_arguments))
    # save training arguments
    with open(osp.join(output_dir, "training_arguments.txt"), "w") as f:
        f.write(str(training_arguments))


def train():
    print("*" * 30 + "Stage 1" + "*" * 30)
    print("Load args...")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    set_seed(42)

    print("*" * 30 + "Stage 2" + "*" * 30)
    print("Load config and model...")
    if training_arguments.resume_from_checkpoint is not None:
        model = LVLMForConditionalGeneration.from_pretrained(training_arguments.resume_from_checkpoint)
    else:
        model_config = LVLMConfig(model_arguments)
        model = LVLMForConditionalGeneration(model_config)
    model.load(model_arguments)  # update config, model, weights

    print("*" * 30 + "Stage 3" + "*" * 30)
    print("Save args and config...")
    save_args(model_arguments, data_arguments, training_arguments, model.config)

    print("*" * 30 + "Stage 4" + "*" * 30)
    print("Load training_recipe...")
    training_recipe = RECIPE_FACTORY[training_arguments.training_recipe](training_arguments)
    model = training_recipe(model)  # tune_type

    if training_arguments.multiloader:
        print("*" * 30 + "Stage 5" + "*" * 30)
        print("Create data_module...")
        with open(training_arguments.ratio_json, "r", encoding="utf-8") as f:
            ratio_dict = json.load(f)
        task_loaders, collator = create_multi_data_module(
            model=model,
            data_arguments=data_arguments,
            ratio_dict=ratio_dict,
            mode="train",
        )

        print("*" * 30 + "Stage 6" + "*" * 30)
        print("Create trainer and train...")
        special_args = SimpleNamespace(
            use_distributed=training_arguments.local_rank != -1,
            dataloader_num_workers=training_arguments.dataloader_num_workers,
            per_device_train_batch_size=training_arguments.per_device_train_batch_size,
            world_size=training_arguments.world_size,
            process_index=training_arguments.process_index,
        )

        trainer = LVLMMULTITrainer(
            model=model,
            tokenizer=model.tokenizer,
            datasets=task_loaders,
            collate_fn=collator,
            special_args=special_args,
            args=training_arguments,
        )
        trainer.train()
    else:
        print("*" * 30 + "Stage 5" + "*" * 30)
        print("Create data_module...")
        data_module = create_data_module(
            model=model,
            data_arguments=data_arguments,
            mode="train",
        )

        print("*" * 30 + "Stage 6" + "*" * 30)
        print("Create trainer and train...")
        trainer = LVLMTrainer(
            model=model,
            tokenizer=model.tokenizer,
            args=training_arguments,
            **data_module,
        )
        trainer.train()

    print("*" * 30 + "Stage 7" + "*" * 30)
    print("Save model...")
    training_recipe.save(model, trainer)


if __name__ == "__main__":
    train()
