import torch
import torch.nn as nn


def get_w(weights, keyword):
    return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}


class Connector(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.connector = None

    def load_pretrained_weights(self):
        if self.config.lvlm_connector_path is not None:
            connector_weights = torch.load(self.config.lvlm_connector_path, map_location="cpu")
            self.connector.load_state_dict(get_w(connector_weights, "connector"))

        for p in self.connector.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.connector(x)
