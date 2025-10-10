import monai.transforms as mtf
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch import distributed as dist
from transformers import PretrainedConfig, PreTrainedModel, BertModel

from lvlm.model.encoder.base import EncoderModel


def medmclip_load_config(lvlm_encoder_image3d_name_or_path, **kwargs):
    lvlm_encoder_image3d_config = MedMCLIPConfig.from_pretrained(
        lvlm_encoder_image3d_name_or_path,
        trust_remote_code=True,
    )
    lvlm_encoder_image3d_config.image_size = [
        lvlm_encoder_image3d_config.frame_size,
        lvlm_encoder_image3d_config.image_size,
        lvlm_encoder_image3d_config.image_size,
    ]
    lvlm_encoder_image3d_config.patch_size = [
        lvlm_encoder_image3d_config.frame_patch_size,
        lvlm_encoder_image3d_config.image_patch_size,
        lvlm_encoder_image3d_config.image_patch_size,
    ]
    return lvlm_encoder_image3d_config


class Image3DProcessor:
    def __init__(self):
        self.mean = 55.6982188356434
        self.std = 66.1943479570918

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
        image3d = (image3d - self.mean) / self.std
        image3d = image3d[np.newaxis, ...]

        if mode == "train":
            image3d = self.transform_train(image3d)
        else:
            image3d = self.transform_val(image3d)
        return image3d


class MedMCLIPModel(EncoderModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.processor = Image3DProcessor()
        self.encoder = MedMCLIP._from_config(config)
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


class MedMCLIPConfig(PretrainedConfig):
    model_type = "medm-pretrain"
    def __init__(
        self,
        args=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if args is not None:
            self.hidden_size = 768
            self.dropout_rate = 0.0
            self.gather_loss = True
            self.local_loss = False

            self.image_size = 256
            self.image_patch_size = 16
            self.frame_size = 32
            self.frame_patch_size = 4
            self.image_num_heads = 12
            self.image_mlp_dim = 3072
            self.image_num_layers = 12

            self.language_model_name_or_path = "google-bert/bert-base-uncased"


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width, pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,  # 注意这里应该用 heads 而不是 dim_head
                dropout=dropout,
                dim_feedforward=mlp_dim,
                batch_first=True
            ) for _ in range(depth)
        ])

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # x = self.transformer(x)
        hidden_states_out = []
        for layer in self.transformer_layers:
            x = layer(x)
            hidden_states_out.append(x.detach())

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return self.mlp_head(x), hidden_states_out


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=True,
    rank=0,
    world_size=1,
):
    if gather_with_grad:
        all_image_features = torch.cat(dist.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(dist.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class MedMCLIP(PreTrainedModel):
    config_class = MedMCLIPConfig
        
    def __init__(self, config):
        super().__init__(config)
        
        self.vision_encoder = ViT(
            image_size = config.image_size, 
            image_patch_size = config.image_patch_size, 
            frames = config.frame_size, 
            frame_patch_size = config.frame_patch_size, 
            num_classes = config.hidden_size, 
            dim = config.hidden_size, 
            depth = config.image_num_layers, 
            heads = config.image_num_heads, 
            mlp_dim = config.image_mlp_dim,
            channels = 1,
        )

        self.text_encoder = BertModel.from_pretrained(config.language_model_name_or_path)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.gather_loss = config.gather_loss
        self.local_loss = config.local_loss

    def forward(self, images, input_ids, attention_mask, labels, **kwargs):
        image_features, _ = self.vision_encoder(images)

        text_features = self.text_encoder(input_ids, attention_mask).pooler_output

        if self.gather_loss:
            all_image_features, all_text_features = gather_features(image_features, text_features)
            if self.local_loss:
                logits_per_image = self.logit_scale * image_features @ all_text_features.T
                logits_per_text = self.logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = self.logit_scale * image_features @ text_features.T
            logits_per_text = self.logit_scale * text_features @ image_features.T

        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {
            "loss": loss,
            "logits": (logits_per_image + logits_per_text) / 2.0,
        }
