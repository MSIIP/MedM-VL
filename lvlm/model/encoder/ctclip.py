import monai.transforms as mtf
import numpy as np
import torch
from transformers import AutoConfig
from transformer_maskgit import CTViT

from lvlm.model.encoder.base import EncoderModel


def ctclip_load_config(lvlm_encoder_image3d_name_or_path, **kwargs):
    lvlm_encoder_image3d_config = AutoConfig.from_pretrained(
        lvlm_encoder_image3d_name_or_path,
        trust_remote_code=True,
    )
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


class CTCLIPModel(EncoderModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.processor = Image3DProcessor()
        self.encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )
        self.encoder.cpu().eval()

    def load_pretrained_weights(self):
        self.encoder.load(config.encoder_image3d_name_or_path)
        self.encoder.cpu().eval()

    def forward(self, x, select_layer, select_feature):
        features = self.encoder(x, return_encoded_tokens=True) 
        features = features.view(features.shape[0], -1, features.shape[-1])
        return features


if __name__ == "__main__":
    config = AutoConfig.from_pretrained("/home/shiym/work_dirs/MedM-VL/MedM-VL-CT-3B-medmclip-finetune")
    config.encoder_image3d_name_or_path = "/hdd/shiym/work_dirs/MedM-VL/CT-CLIP_v2_visual_transformer.pt"
    model = CTCLIPModel(config)

    image = np.load("/home/shiym/datasets/medical-image-analysis/CT-RATE/dataset/preprocessed/valid/valid_1/valid_1_a/valid_1_a_1.npy")
    resize_func = mtf.Resize(spatial_size=[240, 480, 480], mode="bilinear")
    image = resize_func(torch.tensor(image).unsqueeze(0)).unsqueeze(0)  # (B, C, D, H, W)

    image = model.encoder(image, return_encoded_tokens=True)  # (B, 24, 24, 24, 512)
    # 模型部分参数被写死在cuda上，导致可能无法实现多卡训练，或者只能用ZeRO-2策略？

    print("done")
