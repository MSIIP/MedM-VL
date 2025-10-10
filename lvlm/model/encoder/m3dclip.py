import monai.transforms as mtf
import numpy as np
import torch
from transformers import AutoConfig, AutoModel

from lvlm.model.encoder.base import EncoderModel


def m3dclip_load_config(lvlm_encoder_image3d_name_or_path, **kwargs):
    lvlm_encoder_image3d_config = AutoConfig.from_pretrained(
        lvlm_encoder_image3d_name_or_path,
        trust_remote_code=True,
    )
    lvlm_encoder_image3d_config.image_size = lvlm_encoder_image3d_config.img_size
    return lvlm_encoder_image3d_config


class Image3DProcessor:
    def __init__(self):
        self.transform_train = mtf.Compose(
            [
                mtf.CropForeground(),
                mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear"),
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        self.transform_val = mtf.Compose(
            [
                mtf.CropForeground(),
                mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear"),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

    def __call__(self, image3d, mode):
        image3d = image3d.astype(np.float32) / 255.0
        image3d = image3d[np.newaxis, ...]
        image3d = image3d - image3d.min()
        image3d = image3d / np.clip(image3d.max(), a_min=1e-8, a_max=None)

        if mode == "train":
            image3d = self.transform_train(image3d)
        else:
            image3d = self.transform_val(image3d)
        return image3d


class M3dclipModel(EncoderModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.processor = Image3DProcessor()
        self.encoder = AutoModel.from_config(config)
        self.encoder.requires_grad_(False)

    def load_pretrained_weights(self):
        self.encoder = self.encoder.from_pretrained(
            self.config.lvlm_encoder_image3d_name_or_path,
            trust_remote_code=True,
        )
        self.encoder.requires_grad_(False)

    def forward(self, x, select_layer, select_feature):
        _, features = self.encoder.vision_encoder(x)
        features = features[select_layer]

        if select_feature == "patch":
            features = features[:, 1:]
        elif select_feature == "cls_patch":
            features = features

        return features
