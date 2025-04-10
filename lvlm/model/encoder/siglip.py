from PIL import Image

import monai.transforms as mtf
import numpy as np
import torch
from transformers import AutoConfig
from transformers import SiglipImageProcessor, SiglipVisionModel

from lvlm.model.encoder.base import EncoderModel


def siglip_load_config(model_arguments):
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
        self.processor = SiglipImageProcessor.from_pretrained(
            config.encoder_image_name_or_path,
            cache_dir=config.cache_dir_hf,
        )

        self.transform_3d_train = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                # mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                # mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        self.transform_3d_val = mtf.Compose([mtf.ToTensor(dtype=torch.float)])

    def __call__(self, image, mode):
        if isinstance(image, Image.Image):  # process 2D image
            image = self.processor(image, return_tensors="pt")
            image = image["pixel_values"][0]  # [3, H, W]
        elif isinstance(image, np.ndarray):  # process 3D image
            if mode == "train":
                image = self.transform_3d_train(image)
            else:
                image = self.transform_3d_val(image)  # [1, D, H, W], [0, 255]
            image = image.to(dtype=torch.uint8)
            image = image.permute(1, 0, 2, 3)  # [D, 1, H, W]
            image = image.expand(-1, 3, -1, -1)  # [D, 3, H, W]
            image = self.processor(image, return_tensors="pt")  # [D, 3, H, W]
            image = image["pixel_values"].permute(1, 0, 2, 3)  # [3, D, H, W]

        return image


class SiglipModel(EncoderModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config.encoder_image_config
        self.processor = ImageProcessor(config)
        self.encoder = SiglipVisionModel.from_pretrained(
            config.encoder_image_name_or_path,
            cache_dir=config.cache_dir_hf,
        )
        self.encoder.requires_grad_(False)
