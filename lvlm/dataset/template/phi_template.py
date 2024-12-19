from dataclasses import dataclass

from lvlm.dataset.template.base import Template
from lvlm.dataset.template.formatter import Formatter, EmptyFormatter, StringFormatter

system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


@dataclass
class PhiTemplate(Template):
    system: "Formatter" = EmptyFormatter(slot=system + " ")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT: " + "{{content}}" + "<|endoftext|>")
    separator: "Formatter" = EmptyFormatter(slot=[" ASSISTANT:", "<|endoftext|>"])


@dataclass
class Phi3Template(Template):
    system: "Formatter" = EmptyFormatter(slot=system + " ")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT: " + "{{content}}" + "<|endoftext|>")
    separator: "Formatter" = EmptyFormatter(slot=[" ASSISTANT:", "<|endoftext|>"])
