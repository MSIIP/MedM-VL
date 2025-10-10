import re

import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from transformers import PretrainedConfig

from lvlm.model.connector.base import Connector

ACT_TYPE = {"relu": nn.ReLU, "gelu": nn.GELU}


class ConnectorConfig(PretrainedConfig):
    def __init__(self, lvlm_llm_config, lvlm_encoder_config, **kwargs):
        if kwargs["lvlm_connector_image_type"] is not None:
            self.lvlm_connector_type = kwargs["lvlm_connector_image_type"]
            self.lvlm_connector_name = kwargs["lvlm_connector_image_name"]
            self.lvlm_connector_path = kwargs["lvlm_connector_image_path"]
        elif kwargs["lvlm_connector_image3d_type"] is not None:
            self.lvlm_connector_type = kwargs["lvlm_connector_image3d_type"]
            self.lvlm_connector_name = kwargs["lvlm_connector_image3d_name"]
            self.lvlm_connector_path = kwargs["lvlm_connector_image3d_path"]

        self.image_size = lvlm_encoder_config.image_size  # list
        self.patch_size = lvlm_encoder_config.patch_size  # list
        self.input_dim = lvlm_encoder_config.hidden_size
        self.output_dim = lvlm_llm_config.hidden_size



def spatial_pooling_load_config(lvlm_llm_config, lvlm_encoder_config, **kwargs):
    return ConnectorConfig(lvlm_llm_config, lvlm_encoder_config, **kwargs)


class SpatialPooling(Connector):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        image_size = config.image_size
        patch_size = config.patch_size
        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.pooling_size = 2
        self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", config.lvlm_connector_name)
        act_type = config.lvlm_connector_name.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.input_dim, config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(config.output_dim, config.output_dim))

        self.connector = nn.Sequential(*modules)
        for p in self.connector.parameters():
            p.requires_grad = False

    def forward(self, x):
        batch_size = x.shape[0]
        embedding_dim = x.shape[-1]

        seq2cube = Rearrange(
            "b (p1 p2 p3) d -> b d p1 p2 p3",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_pre[0],
            p2=self.num_patches_pre[1],
            p3=self.num_patches_pre[2],
        )
        x = seq2cube(x)
        x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
        
        cube2seq = Rearrange(
            "b d p1 p2 p3 -> b (p1 p2 p3) d",
            b=batch_size,
            d=embedding_dim,
            p1=self.num_patches_post[0],
            p2=self.num_patches_post[1],
            p3=self.num_patches_post[2],
        )
        x = cube2seq(x)
        x = self.connector(x)

        return x
