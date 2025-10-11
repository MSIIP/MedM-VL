import re

import torch.nn as nn
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

        self.image_size = lvlm_encoder_config.image_size  # int
        self.patch_size = lvlm_encoder_config.patch_size  # int
        self.input_dim = lvlm_encoder_config.hidden_size
        self.output_dim = lvlm_llm_config.hidden_size


def mlp_load_config(lvlm_llm_config, lvlm_encoder_config, **kwargs):
    return ConnectorConfig(lvlm_llm_config, lvlm_encoder_config, **kwargs)


class MLPConnector(Connector):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

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
