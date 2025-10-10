from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM


def llama_load_config(lvlm_llm_name_or_path, **kwargs):
    lvlm_llm_config = AutoConfig.from_pretrained(
        lvlm_llm_name_or_path,
        trust_remote_code=True,
    )
    return lvlm_llm_config


def llama_load_tokenizer(
    lvlm_llm_name_or_path,
    lvlm_llm_max_length,
    lvlm_llm_padding_side,
):
    tokenizer = AutoTokenizer.from_pretrained(
        lvlm_llm_name_or_path,
        model_max_length=lvlm_llm_max_length,
        padding_side=lvlm_llm_padding_side,
        use_fast=False,
        trust_remote_code=True,
    )
    return tokenizer


def llama_load_model(config):
    model = LlamaForCausalLM._from_config(config)
    model.requires_grad_(False)
    return model


def llama3_load_config(lvlm_llm_name_or_path, **kwargs):
    lvlm_llm_config = AutoConfig.from_pretrained(
        lvlm_llm_name_or_path,
        trust_remote_code=True,
    )
    return lvlm_llm_config


# bos_token: <|begin_of_text|> 128000
# eos_token: <|eot_id|> 128009
# pad_token: None None
# unk_token: None None
def llama3_load_tokenizer(
    lvlm_llm_name_or_path,
    lvlm_llm_max_length,
    lvlm_llm_padding_side,
):
    tokenizer = AutoTokenizer.from_pretrained(
        lvlm_llm_name_or_path,
        model_max_length=lvlm_llm_max_length,
        padding_side=lvlm_llm_padding_side,
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def llama3_load_model(config):
    model = AutoModelForCausalLM.from_config(config)
    model.requires_grad_(False)
    return model
