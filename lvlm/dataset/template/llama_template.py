from dataclasses import dataclass

from lvlm.dataset.template.base import Template
from lvlm.dataset.template.formatter import Formatter, EmptyFormatter, StringFormatter

system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


# bos_token: <s> 1
# eos_token: </s> 2
# pad_token: <unk> 0
# unk_token: <unk> 0
@dataclass
class LlamaTemplate(Template):
    system: "Formatter" = EmptyFormatter(slot=system + " ")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT: " + "{{content}}" + "</s>")
    separator: "Formatter" = EmptyFormatter(slot=[" ASSISTANT:", "</s>"])


# bos_token: <|begin_of_text|> 128000
# eos_token: <|eot_id|> 128009
# pad_token: None None
# unk_token: None None
@dataclass
class Llama3Template(Template):
    system: "Formatter" = EmptyFormatter(slot=system + " ")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT: " + "{{content}}" + "<|eot_id|>")
    separator: "Formatter" = EmptyFormatter(slot=[" ASSISTANT:", "<|eot_id|>"])
