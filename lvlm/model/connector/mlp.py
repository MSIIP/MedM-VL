import re

import torch.nn as nn
from transformers import PretrainedConfig

from lvlm.model.connector.base import Connector

ACT_TYPE = {"relu": nn.ReLU, "gelu": nn.GELU}


class ConnectorConfig(PretrainedConfig):
    def __init__(self, model_arguments, llm_config, encoder_config):
        self.connector_type = model_arguments.connector_image_type
        self.connector_name = model_arguments.connector_image_name
        self.connector_path = model_arguments.connector_image_path
        self.img_size = [encoder_config.image_size] * 2
        self.patch_size = [encoder_config.patch_size] * 2

        self.input_dim = encoder_config.hidden_size
        self.output_dim = llm_config.hidden_size


def mlp_load_config(model_arguments, llm_config, encoder_config):
    return ConnectorConfig(model_arguments, llm_config, encoder_config)


class MLPConnector(Connector):
    def __init__(self, config):
        super().__init__(config)
        config = config.connector_image_config

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", config.connector_name)
        act_type = config.connector_name.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.input_dim, config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(
                nn.Linear(config.output_dim, config.output_dim)
            )

        self.connector = nn.Sequential(*modules)
        for p in self.connector.parameters():
            p.requires_grad = False
