from lvlm.model.llm.llama import llama_load_config, llama_load_model
from lvlm.model.llm.llama import llama3_load_config, llama3_load_model
from lvlm.model.llm.qwen import qwen2_load_config, qwen2_load_model
from lvlm.model.llm.phi import phi_load_config, phi_load_model
from lvlm.model.llm.phi import phi3_load_config, phi3_load_model


LLM_FACTORY = {
    "llama": (llama_load_config, llama_load_model),
    "llama3": (llama3_load_config, llama3_load_model),
    "qwen2": (qwen2_load_config, qwen2_load_model),
    "phi": (phi_load_config, phi_load_model),
    "phi3": (phi3_load_config, phi3_load_model),
}
