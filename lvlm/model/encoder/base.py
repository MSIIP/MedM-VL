import torch.nn as nn


class EncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.processor = None
        self.encoder = None

    def forward(self, x, select_layer, select_feature):
        features = self.encoder(x, output_hidden_states=True)
        features = features.hidden_states[select_layer]

        if select_feature == "patch":
            features = features[:, 1:]
        elif select_feature == "cls_patch":
            features = features

        return features
