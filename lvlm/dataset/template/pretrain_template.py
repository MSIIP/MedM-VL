import copy
from dataclasses import dataclass

from lvlm.dataset.template.base import Template
from lvlm.dataset.template.formatter import Formatter, EmptyFormatter, StringFormatter
from lvlm.utils.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE3D_TOKEN


@dataclass
class PretrainTemplate(Template):
    system: "Formatter" = EmptyFormatter(slot="")
    format_user: "Formatter" = EmptyFormatter(slot=DEFAULT_IMAGE_TOKEN)
    format_assistant: "Formatter" = StringFormatter(slot="{{content}}")
    separator: "Formatter" = EmptyFormatter(slot=["", ""])

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        instruction_len = len(self.tokenizer_special_token(DEFAULT_IMAGE_TOKEN, tokenizer))
        labels[:instruction_len] = IGNORE_INDEX
        return labels


@dataclass
class PretrainImage3dTemplate(Template):
    system: "Formatter" = EmptyFormatter(slot="")
    format_user: "Formatter" = EmptyFormatter(slot=DEFAULT_IMAGE3D_TOKEN)
    format_assistant: "Formatter" = StringFormatter(slot="{{content}}")
    separator: "Formatter" = EmptyFormatter(slot=["", ""])

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        instruction_len = len(self.tokenizer_special_token(DEFAULT_IMAGE3D_TOKEN, tokenizer))
        labels[:instruction_len] = IGNORE_INDEX
        return labels
