from lvlm.dataset.template.pretrain_template import PretrainTemplate, PretrainImage3dTemplate
from lvlm.dataset.template.llama_template import LlamaTemplate, Llama3Template
from lvlm.dataset.template.qwen_template import Qwen2Template
from lvlm.dataset.template.phi_template import PhiTemplate, Phi3Template

TEMPlATE_FACTORY = {
    "pretrain": PretrainTemplate,
    "pretrain_image3d": PretrainImage3dTemplate,
    "llama": LlamaTemplate,
    "llama3": Llama3Template,
    "qwen2": Qwen2Template,
    "phi": PhiTemplate,
    "phi3": Phi3Template,
}
