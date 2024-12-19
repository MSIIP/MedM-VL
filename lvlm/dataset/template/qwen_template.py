from dataclasses import dataclass

from lvlm.dataset.template.base import Template
from lvlm.dataset.template.formatter import Formatter, EmptyFormatter, StringFormatter

system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


# bos_token: None None
# eos_token: <|im_end|> 151645
# pad_token: <|endoftext|> 151643
# unk_token: None None
@dataclass
class Qwen2Template(Template):
    system: "Formatter" = EmptyFormatter(slot=system + " ")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT: " + "{{content}}" + "<|im_end|>")
    separator: "Formatter" = EmptyFormatter(slot=[" ASSISTANT:", "<|im_end|>"])
