from transformers import AutoConfig
from transformers import CLIPImageProcessor, CLIPVisionModel

from lvlm.model.encoder.base import EncoderModel


def clip_load_config(model_arguments):
    encoder_image_config = AutoConfig.from_pretrained(
        model_arguments.encoder_image_name_or_path,
        cache_dir=model_arguments.cache_dir_hf,
        trust_remote_code=True,
    )
    if hasattr(encoder_image_config, "vision_config"):
        _name_or_path = encoder_image_config._name_or_path
        encoder_image_config = getattr(encoder_image_config, "vision_config", encoder_image_config)
        encoder_image_config._name_or_path = _name_or_path

    return encoder_image_config


class ImageProcessor:
    def __init__(self, config):
        self.processor = CLIPImageProcessor.from_pretrained(
            config.encoder_image_name_or_path,
            cache_dir=config.cache_dir_hf,
        )

    def __call__(self, image, mode):
        image = self.processor(image, return_tensors="pt")
        image = image["pixel_values"][0]
        return image


class ClipModel(EncoderModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config.encoder_image_config
        self.processor = ImageProcessor(config)
        self.encoder = CLIPVisionModel.from_pretrained(
            config.encoder_image_name_or_path,
            cache_dir=config.cache_dir_hf,
        )
        self.encoder.requires_grad_(False)
