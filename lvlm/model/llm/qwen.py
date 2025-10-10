from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def qwen2_load_config(lvlm_llm_name_or_path, **kwargs):
    lvlm_llm_config = AutoConfig.from_pretrained(
        lvlm_llm_name_or_path,
        trust_remote_code=True,
    )
    return lvlm_llm_config


# bos_token: None None
# eos_token: <|im_end|> 151645
# pad_token: <|endoftext|> 151643
# unk_token: None None
def qwen2_load_tokenizer(
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


def qwen2_load_model(config):
    model = AutoModelForCausalLM.from_config(config)
    model.requires_grad_(False)
    return model
