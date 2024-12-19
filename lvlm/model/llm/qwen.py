from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def qwen2_load_config(model_arguments):
    llm_config = AutoConfig.from_pretrained(
        model_arguments.llm_name_or_path,
        cache_dir=model_arguments.cache_dir_hf,
        trust_remote_code=True,
    )
    return llm_config


# bos_token: None None
# eos_token: <|im_end|> 151645
# pad_token: <|endoftext|> 151643
# unk_token: None None
def qwen2_load_model(config):
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

    return model, tokenizer
