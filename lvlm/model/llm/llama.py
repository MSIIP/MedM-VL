from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM


def llama_load_config(model_arguments):
    llm_config = AutoConfig.from_pretrained(
        model_arguments.llm_name_or_path,
        cache_dir=model_arguments.cache_dir_hf,
        trust_remote_code=True,
    )
    return llm_config


def llama_load_model(config):
    model = LlamaForCausalLM.from_pretrained(
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


def llama3_load_config(model_arguments):
    llm_config = AutoConfig.from_pretrained(
        model_arguments.llm_name_or_path,
        cache_dir=model_arguments.cache_dir_hf,
        trust_remote_code=True,
    )
    return llm_config


# bos_token: <|begin_of_text|> 128000
# eos_token: <|eot_id|> 128009
# pad_token: None None
# unk_token: None None
def llama3_load_model(config):
    model = AutoModelForCausalLM.from_pretrained(
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
