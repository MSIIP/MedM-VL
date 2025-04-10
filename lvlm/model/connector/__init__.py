from lvlm.model.connector.mlp import mlp_load_config, MLPConnector
from lvlm.model.connector.spatial_pooling import spatial_pooling_load_config, SpatialPooling
from lvlm.model.connector.attn_pooling import attn_pooling_load_config, AttnPooling

CONNECTOR_FACTORY = {
    "mlp": (mlp_load_config, MLPConnector),
    "spatial_pooling": (spatial_pooling_load_config, SpatialPooling),
    "attn_pooling": (attn_pooling_load_config, AttnPooling),
}