from transformers import AutoConfig
from transformers import CLIPImageProcessor, CLIPVisionModel

from lvlm.model.encoder.base import EncoderModel


def clip_load_config(lvlm_encoder_image_name_or_path, **kwargs):
    lvlm_encoder_image_config = AutoConfig.from_pretrained(
        lvlm_encoder_image_name_or_path,
        trust_remote_code=True,
    )
    if hasattr(lvlm_encoder_image_config, "vision_config"):
        _name_or_path = lvlm_encoder_image_config._name_or_path
        lvlm_encoder_image_config = getattr(lvlm_encoder_image_config, "vision_config", lvlm_encoder_image_config)
        lvlm_encoder_image_config._name_or_path = _name_or_path

    lvlm_encoder_image_config.lvlm_encoder_image_name_or_path = lvlm_encoder_image_name_or_path
    return lvlm_encoder_image_config


class ImageProcessor:
    def __init__(self, config):
        self.processor = CLIPImageProcessor.from_pretrained(config.lvlm_encoder_image_name_or_path)

    def __call__(self, image, mode):
        image = self.processor(image, return_tensors="pt")
        image = image["pixel_values"][0]
        return image


class ClipModel(EncoderModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.processor = ImageProcessor(config)
        self.encoder = CLIPVisionModel._from_config(config)
        self.encoder.requires_grad_(False)

    def load_pretrained_weights(self):
        self.encoder = self.encoder.from_pretrained(self.config.lvlm_encoder_image_name_or_path)
        self.encoder.requires_grad_(False)
