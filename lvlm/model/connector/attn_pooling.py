import re

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from lvlm.model.connector.base import Connector


ACT_TYPE = {"relu": nn.ReLU, "gelu": nn.GELU}


class ConnectorConfig(PretrainedConfig):
    def __init__(self, model_arguments, llm_config, encoder_config):
        if model_arguments.connector_image_type is not None:
            self.connector_type = model_arguments.connector_image_type
            self.connector_name = model_arguments.connector_image_name
            self.connector_path = model_arguments.connector_image_path
        elif model_arguments.connector_image3d_type is not None:
            self.connector_type = model_arguments.connector_image3d_type
            self.connector_name = model_arguments.connector_image3d_name
            self.connector_path = model_arguments.connector_image3d_path
        self.image_size = encoder_config.image_size
        self.patch_size = encoder_config.patch_size

        self.input_dim = encoder_config.hidden_size
        self.output_dim = llm_config.hidden_size


def attn_pooling_load_config(model_arguments, llm_config, encoder_config):
    return ConnectorConfig(model_arguments, llm_config, encoder_config)


class AttnPooling(Connector):
    def __init__(self, config):
        super().__init__(config)
        if config.connector_image_config is not None:
            config = config.connector_image_config
        elif config.connector_image3d_config is not None:
            config = config.connector_image3d_config

        d_model = config.input_dim
        n_queries = (config.image_size // config.patch_size) ** 2 
        n_head = 8

        self.n_queries = n_queries
        self.query = nn.Parameter(torch.randn(n_queries, d_model))  # 可学习的查询向量

        # MultiheadAttention的配置
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        # self.to_out = nn.Linear(d_model, d_model, bias=False)  # 输出映射

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", config.connector_name)
        act_type = config.connector_name.split("_")[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.input_dim, config.output_dim)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(config.output_dim, config.output_dim))

        self.connector = nn.Sequential(*modules)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        batch_size, num_tokens, d_model = x.shape  # x: (batch_size, num_tokens, d_model)
        
        # 构造查询向量
        q = self.query.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, n_queries, d_model)

        # 使用MultiheadAttention计算注意力
        # 注意: MultiheadAttention需要输入q, k, v的维度是(batch_size, seq_len, embed_dim)
        x, _ = self.attn(q, x, x)  # attn_output: (batch_size, n_queries, d_model)
        x = self.connector(x)  # (batch_size, n_queries, d_model)

        return x
