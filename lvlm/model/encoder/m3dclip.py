import monai.transforms as mtf
import torch
from transformers import AutoConfig, AutoModel

from lvlm.model.encoder.base import EncoderModel


def m3dclip_load_config(model_arguments):
    encoder_image3d_config = AutoConfig.from_pretrained(
        model_arguments.encoder_image3d_name_or_path,
        cache_dir=model_arguments.cache_dir_hf,
        trust_remote_code=True,
    )
    return encoder_image3d_config


class Image3DProcessor:
    def __init__(self):
        self.transform_train = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        self.transform_val = mtf.Compose([mtf.ToTensor(dtype=torch.float)])

    def __call__(self, image3d, mode):
        if mode == "train":
            image3d = self.transform_train(image3d)
        else:
            image3d = self.transform_val(image3d)
        return image3d


class M3dclipModel(EncoderModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config.encoder_image3d_config
        self.processor = Image3DProcessor()
        self.encoder = AutoModel.from_pretrained(
            config.encoder_image3d_name_or_path,
            cache_dir=config.cache_dir_hf,
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
