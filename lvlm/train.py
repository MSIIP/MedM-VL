import json
import random
import os
import os.path as osp
from types import SimpleNamespace

import transformers

from lvlm.utils.arguments import ModelArguments, TrainingArguments, set_seed
from lvlm.dataset.dataset import create_data_module, create_multi_data_module
from lvlm.model.configuration_lvlm import LVLMConfig
from lvlm.model.modeling_lvlm import LVLMForConditionalGeneration
from lvlm.utils.training_recipe import RECIPE_FACTORY
from lvlm.utils.trainer_lvlm import LVLMTrainer, LVLMMULTITrainer


def train():
    print("*" * 30 + "Stage 1" + "*" * 30)
    print("Load args...")
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(42)

    print("*" * 30 + "Stage 2" + "*" * 30)
    print("Load config and model...")
    if training_args.resume_from_checkpoint is not None:
        model = LVLMForConditionalGeneration.from_pretrained(training_args.resume_from_checkpoint)
    else:
        config = LVLMConfig(
            lvlm_llm_type=model_args.lvlm_llm_type,
            lvlm_llm_name_or_path=model_args.lvlm_llm_name_or_path,
            lvlm_llm_max_length=model_args.lvlm_llm_max_length,
            lvlm_llm_padding_side=model_args.lvlm_llm_padding_side,
            lvlm_llm_attn_implementation=model_args.lvlm_llm_attn_implementation,
            lvlm_encoder_image_type=model_args.lvlm_encoder_image_type,
            lvlm_encoder_image_name_or_path=model_args.lvlm_encoder_image_name_or_path,
            lvlm_encoder_image_select_layer=model_args.lvlm_encoder_image_select_layer,
            lvlm_encoder_image_select_feature=model_args.lvlm_encoder_image_select_feature,
            lvlm_connector_image_type=model_args.lvlm_connector_image_type,
            lvlm_connector_image_name=model_args.lvlm_connector_image_name,
            lvlm_connector_image_path=model_args.lvlm_connector_image_path,
            lvlm_encoder_image3d_type=model_args.lvlm_encoder_image3d_type,
            lvlm_encoder_image3d_name_or_path=model_args.lvlm_encoder_image3d_name_or_path,
            lvlm_encoder_image3d_select_layer=model_args.lvlm_encoder_image3d_select_layer,
            lvlm_encoder_image3d_select_feature=model_args.lvlm_encoder_image3d_select_feature,
            lvlm_connector_image3d_type=model_args.lvlm_connector_image3d_type,
            lvlm_connector_image3d_name=model_args.lvlm_connector_image3d_name,
            lvlm_connector_image3d_path=model_args.lvlm_connector_image3d_path,
        )
        model = LVLMForConditionalGeneration(config)
        model.load_pretrained_weights()  # load pretrained weights

    print("*" * 30 + "Stage 3" + "*" * 30)
    print("Load training_recipe...")
    training_recipe = RECIPE_FACTORY[training_args.training_recipe](training_args)
    model = training_recipe(model)  # tune_type

    if training_args.multiloader:
        print("*" * 30 + "Stage 4" + "*" * 30)
        print("Create data_module...")
        with open(training_args.data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        with open(training_args.ratio_json, "r", encoding="utf-8") as f:
            ratio_dict = json.load(f)
        task_loaders, collator = create_multi_data_module(
            all_data=all_data,
            model=model,
            data_arguments=training_args,
            ratio_dict=ratio_dict,
            mode="train",
        )

        print("*" * 30 + "Stage 5" + "*" * 30)
        print("Create trainer and train...")
        special_args = SimpleNamespace(
            use_distributed=training_args.local_rank != -1,
            dataloader_num_workers=training_args.dataloader_num_workers,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            world_size=training_args.world_size,
            process_index=training_args.process_index,
        )

        trainer = LVLMMULTITrainer(
            model=model,
            tokenizer=model.tokenizer,
            datasets=task_loaders,
            collate_fn=collator,
            special_args=special_args,
            args=training_args,
        )
        trainer.train()

    else:
        print("*" * 30 + "Stage 4" + "*" * 30)
        print("Create data_module...")
        with open(training_args.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_module = create_data_module(
            data=data,
            conv_version=training_args.conv_version,
            image_dir=training_args.image_dir,
            image3d_dir=training_args.image3d_dir,
            model=model,
            mode="train",
        )

        print("*" * 30 + "Stage 5" + "*" * 30)
        print("Create trainer and train...")
        trainer = LVLMTrainer(
            model=model,
            args=training_args,
            train_dataset=data_module["train_dataset"],
            data_collator=data_module["data_collator"],
            tokenizer=model.tokenizer,
        )
        trainer.train()

    print("*" * 30 + "Stage 6" + "*" * 30)
    print("Save model...")
    training_recipe.save(model, trainer)


if __name__ == "__main__":
    train()
