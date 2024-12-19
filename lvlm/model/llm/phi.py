from transformers import AutoConfig, AutoTokenizer
from transformers import PhiForCausalLM
from transformers import Phi3ForCausalLM


def phi_load_config(model_arguments):
    llm_config = AutoConfig.from_pretrained(
        model_arguments.llm_name_or_path,
        cache_dir=model_arguments.cache_dir_hf,
        trust_remote_code=True,
    )
    return llm_config


def phi_load_model(config):
    model = PhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.llm_name_or_path,
        attn_implementation=config.llm_attn_implementation,
        cache_dir=config.cache_dir_hf,
    )
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_name_or_path,
        model_max_length=config.llm_max_length,
        padding_side=config.llm_padding_side,
        cache_dir=config.cache_dir_hf,
        use_fast=config.tokenizer_use_fast,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    return model, tokenizer


def phi3_load_config(model_arguments):
    llm_config = AutoConfig.from_pretrained(
        model_arguments.llm_name_or_path,
        cache_dir=model_arguments.cache_dir_hf,
        trust_remote_code=True,
    )
    return llm_config


def phi3_load_model(config):
    model = Phi3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.llm_name_or_path,
        attn_implementation=config.llm_attn_implementation,
        cache_dir=config.cache_dir_hf,
    )
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_name_or_path,
        model_max_length=config.llm_max_length,
        padding_side=config.llm_padding_side,
        cache_dir=config.cache_dir_hf,
        use_fast=config.tokenizer_use_fast,
        trust_remote_code=True,
    )

    return model, tokenizer
