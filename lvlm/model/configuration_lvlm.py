from types import SimpleNamespace

from transformers import PretrainedConfig

from lvlm.model.llm import LLM_FACTORY
from lvlm.model.encoder import ENCODER_FACTORY
from lvlm.model.connector import CONNECTOR_FACTORY


class LVLMConfig(PretrainedConfig):
    def __init__(self, model_arguments=None, **kwargs):
        if model_arguments is not None:
            self.cache_dir_hf = model_arguments.cache_dir_hf

            self.llm_type = model_arguments.llm_type
            self.llm_name_or_path = model_arguments.llm_name_or_path
            self.llm_max_length = model_arguments.llm_max_length
            self.llm_padding_side = model_arguments.llm_padding_side
            self.llm_attn_implementation = model_arguments.llm_attn_implementation
            self.tokenizer_use_fast = model_arguments.tokenizer_use_fast

            self.encoder_image_type = model_arguments.encoder_image_type
            self.encoder_image_name_or_path = model_arguments.encoder_image_name_or_path
            self.encoder_image_select_layer = model_arguments.encoder_image_select_layer
            self.encoder_image_select_feature = model_arguments.encoder_image_select_feature
            self.connector_image_type = model_arguments.connector_image_type
            self.connector_image_name = model_arguments.connector_image_name
            self.connector_image_path = model_arguments.connector_image_path

            self.encoder_image3d_type = model_arguments.encoder_image3d_type
            self.encoder_image3d_name_or_path = model_arguments.encoder_image3d_name_or_path
            self.encoder_image3d_select_layer = model_arguments.encoder_image3d_select_layer
            self.encoder_image3d_select_feature = model_arguments.encoder_image3d_select_feature
            self.connector_image3d_type = model_arguments.connector_image3d_type
            self.connector_image3d_name = model_arguments.connector_image3d_name
            self.connector_image3d_path = model_arguments.connector_image3d_path

            # modules
            self.llm_config = LLM_FACTORY[self.llm_type][0](model_arguments)

            if self.encoder_image_type is not None:
                self.encoder_image_config = ENCODER_FACTORY[self.encoder_image_type][0](model_arguments)
                self.connector_image_config = CONNECTOR_FACTORY[self.connector_image_type][0](
                    model_arguments=model_arguments,
                    llm_config=self.llm_config,
                    encoder_config=self.encoder_image_config,
                )
            else:
                self.encoder_image_config = None
                self.connector_image_config = None

            if self.encoder_image3d_type is not None:
                self.encoder_image3d_config = ENCODER_FACTORY[self.encoder_image3d_type][0](model_arguments)
                self.connector_image3d_config = CONNECTOR_FACTORY[self.connector_image3d_type][0](
                    model_arguments=model_arguments,
                    llm_config=self.llm_config,
                    encoder_config=self.encoder_image3d_config,
                )
            else:
                self.encoder_image3d_config = None
                self.connector_image3d_config = None

            self.hidden_size = self.llm_config.hidden_size  # only for deepspeed

        super().__init__(**kwargs)  # AttributeError: 'LVLMConfig' object has no attribute 'pruned_heads'

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        model_arguments = SimpleNamespace(**config_dict)
        config_dict["llm_config"] = LLM_FACTORY[model_arguments.llm_type][0](model_arguments)
        if model_arguments.encoder_image_type is not None:
            config_dict["encoder_image_config"] = ENCODER_FACTORY[model_arguments.encoder_image_type][0](model_arguments)
            config_dict["connector_image_config"] = CONNECTOR_FACTORY[model_arguments.connector_image_type][0](
                model_arguments=model_arguments,
                llm_config=config_dict["llm_config"],
                encoder_config=config_dict["encoder_image_config"],
            )
        if model_arguments.encoder_image3d_type is not None:
            config_dict["encoder_image3d_config"] = ENCODER_FACTORY[model_arguments.encoder_image3d_type][0](model_arguments)
            config_dict["connector_image3d_config"] = CONNECTOR_FACTORY[model_arguments.connector_image3d_type][0](
                model_arguments=model_arguments,
                llm_config=config_dict["llm_config"],
                encoder_config=config_dict["encoder_image3d_config"],
            )

        return cls.from_dict(config_dict, **kwargs)