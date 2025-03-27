from lvlm.model.encoder.clip import clip_load_config, ClipModel
from lvlm.model.encoder.siglip import siglip_load_config, SiglipModel
from lvlm.model.encoder.m3dclip import m3dclip_load_config, M3dclipModel
from lvlm.model.encoder.medmclip import medmclip_load_config, MedMCLIPModel


ENCODER_FACTORY = {
    "clip": (clip_load_config, ClipModel),
    "siglip": (siglip_load_config, SiglipModel),
    "m3dclip": (m3dclip_load_config, M3dclipModel),
    "medmclip": (medmclip_load_config, MedMCLIPModel),
}
