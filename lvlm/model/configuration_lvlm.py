from transformers import PretrainedConfig

from lvlm.model.llm import LLM_FACTORY


class LVLMConfig(PretrainedConfig):
    model_type = "lvlm"

    def __init__(
        self,
        lvlm_llm_type: str = None,
        lvlm_llm_name_or_path: str = None,
        lvlm_llm_max_length: int = None,
        lvlm_llm_padding_side: str = None,
        lvlm_llm_attn_implementation: str = None,
        lvlm_encoder_image_type: str = None,
        lvlm_encoder_image_name_or_path: str = None,
        lvlm_encoder_image_select_layer: int = None,
        lvlm_encoder_image_select_feature: str = None,
        lvlm_connector_image_type: str = None,
        lvlm_connector_image_name: str = None,
        lvlm_connector_image_path: str = None,
        lvlm_encoder_image3d_type: str = None,
        lvlm_encoder_image3d_name_or_path: str = None,
        lvlm_encoder_image3d_select_layer: int = None,
        lvlm_encoder_image3d_select_feature: str = None,
        lvlm_connector_image3d_type: str = None,
        lvlm_connector_image3d_name: str = None,
        lvlm_connector_image3d_path: str = None,
        **kwargs,
    ):
        self.lvlm_llm_type = lvlm_llm_type
        self.lvlm_llm_name_or_path = lvlm_llm_name_or_path
        self.lvlm_llm_max_length = lvlm_llm_max_length
        self.lvlm_llm_padding_side = lvlm_llm_padding_side
        self.lvlm_llm_attn_implementation = lvlm_llm_attn_implementation

        self.lvlm_encoder_image_type = lvlm_encoder_image_type
        self.lvlm_encoder_image_name_or_path = lvlm_encoder_image_name_or_path
        self.lvlm_encoder_image_select_layer = lvlm_encoder_image_select_layer
        self.lvlm_encoder_image_select_feature = lvlm_encoder_image_select_feature
        self.lvlm_connector_image_type = lvlm_connector_image_type
        self.lvlm_connector_image_name = lvlm_connector_image_name
        self.lvlm_connector_image_path = lvlm_connector_image_path

        self.lvlm_encoder_image3d_type = lvlm_encoder_image3d_type
        self.lvlm_encoder_image3d_name_or_path = lvlm_encoder_image3d_name_or_path
        self.lvlm_encoder_image3d_select_layer = lvlm_encoder_image3d_select_layer
        self.lvlm_encoder_image3d_select_feature = lvlm_encoder_image3d_select_feature
        self.lvlm_connector_image3d_type = lvlm_connector_image3d_type
        self.lvlm_connector_image3d_name = lvlm_connector_image3d_name
        self.lvlm_connector_image3d_path = lvlm_connector_image3d_path

        if lvlm_llm_type is not None:
            lvlm_llm_config = LLM_FACTORY[lvlm_llm_type][0](
                lvlm_llm_name_or_path=lvlm_llm_name_or_path,
            )
            self.lvlm_llm_initializer_range = lvlm_llm_config.initializer_range  # model._init_weights()
            self.hidden_size = lvlm_llm_config.hidden_size  # only for deepspeed

        super().__init__(**kwargs)  # AttributeError: 'LVLMConfig' object has no attribute 'pruned_heads'
