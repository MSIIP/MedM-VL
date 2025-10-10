from transformers import AutoConfig, AutoTokenizer
from transformers import PhiForCausalLM
from transformers import Phi3ForCausalLM


def phi_load_config(lvlm_llm_name_or_path, **kwargs):
    lvlm_llm_config = AutoConfig.from_pretrained(
        lvlm_llm_name_or_path,
        trust_remote_code=True,
    )
    return lvlm_llm_config


def phi_load_tokenizer(
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
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    return tokenizer


def phi_load_model(config):
    model = PhiForCausalLM._from_config(config)
    model.requires_grad_(False)
    return model


def phi3_load_config(lvlm_llm_name_or_path, **kwargs):
    lvlm_llm_config = AutoConfig.from_pretrained(
        lvlm_llm_name_or_path,
        trust_remote_code=True,
    )
    return lvlm_llm_config


def phi3_load_tokenizer(
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


def phi3_load_model(config):
    model = Phi3ForCausalLM._from_config(config)
    model.requires_grad_(False)
    return model
