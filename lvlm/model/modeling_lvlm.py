import torch
import torch.nn as nn
from transformers import PreTrainedModel

from lvlm.model.configuration_lvlm import LVLMConfig
from lvlm.model.llm import LLM_FACTORY
from lvlm.model.encoder import ENCODER_FACTORY
from lvlm.model.connector import CONNECTOR_FACTORY
from lvlm.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_ID, IMAGE3D_TOKEN_ID


class LVLMPreTrainedModel(PreTrainedModel):
    config_class = LVLMConfig  # from_pretrained
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.llm_config.initializer_range

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LVLMForConditionalGeneration(LVLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.llm, self.tokenizer = LLM_FACTORY[config.llm_type][1](config)

        if config.encoder_image_type is not None:
            self.encoder_image = ENCODER_FACTORY[config.encoder_image_type][1](config)
            self.connector_image = CONNECTOR_FACTORY[config.connector_image_type][1](config)
        else:
            self.encoder_image = None
            self.connector_image = None

        if config.encoder_image3d_type is not None:
            self.encoder_image3d = ENCODER_FACTORY[config.encoder_image3d_type][1](config)
            self.connector_image3d = CONNECTOR_FACTORY[config.connector_image3d_type][1](config)
        else:
            self.encoder_image3d = None
            self.connector_image3d = None

        self.post_init()

    def load(self, model_arguments):
        if model_arguments.llm_type is not None:
            # 临时换LLM可能导致connector维度的不匹配，请谨慎
            # update config
            self.config.llm_type = model_arguments.llm_type
            self.config.llm_name_or_path = model_arguments.llm_name_or_path
            self.config.llm_max_length = model_arguments.llm_max_length
            self.config.llm_padding_side = model_arguments.llm_padding_side
            self.config.llm_attn_implementation = model_arguments.llm_attn_implementation
            self.config.llm_config = LLM_FACTORY[model_arguments.llm_type][0](model_arguments)
            self.config.tokenizer_use_fast = model_arguments.tokenizer_use_fast
            self.config.hidden_size = self.config.llm_config.hidden_size

            # update model
            self.llm, self.tokenizer = LLM_FACTORY[model_arguments.llm_type][1](self.config)

        if model_arguments.encoder_image_type is not None:
            # update config
            self.config.encoder_image_type = model_arguments.encoder_image_type
            self.config.encoder_image_name_or_path = model_arguments.encoder_image_name_or_path
            self.config.encoder_image_select_layer = model_arguments.encoder_image_select_layer
            self.config.encoder_image_select_feature = model_arguments.encoder_image_select_feature
            self.config.connector_image_type = model_arguments.connector_image_type
            self.config.connector_image_name = model_arguments.connector_image_name
            self.config.connector_image_path = model_arguments.connector_image_path
            self.config.encoder_image_config = ENCODER_FACTORY[model_arguments.encoder_image_type][0](model_arguments)
            self.config.connector_image_config = CONNECTOR_FACTORY[model_arguments.connector_image_type][0](
                model_arguments=model_arguments,
                llm_config=self.config.llm_config,
                encoder_config=self.config.encoder_image_config,
            )

            # update model
            self.encoder_image = ENCODER_FACTORY[self.config.encoder_image_type][1](self.config)
            self.connector_image = CONNECTOR_FACTORY[self.config.connector_image_type][1](self.config)
            self.connector_image.load_model(self.config.connector_image_path)

        if model_arguments.encoder_image3d_type is not None:
            # update config
            self.config.encoder_image3d_type = model_arguments.encoder_image3d_type
            self.config.encoder_image3d_name_or_path = model_arguments.encoder_image3d_name_or_path
            self.config.encoder_image3d_select_layer = model_arguments.encoder_image3d_select_layer
            self.config.encoder_image3d_select_feature = model_arguments.encoder_image3d_select_feature
            self.config.connector_image3d_type = model_arguments.connector_image3d_type
            self.config.connector_image3d_name = model_arguments.connector_image3d_name
            self.config.connector_image3d_path = model_arguments.connector_image3d_path
            self.config.encoder_image3d_config = ENCODER_FACTORY[model_arguments.encoder_image3d_type][0](model_arguments)
            self.config.connector_image3d_config = CONNECTOR_FACTORY[model_arguments.connector_image3d_type][0](
                model_arguments=model_arguments,
                llm_config=self.config.llm_config,
                encoder_config=self.config.encoder_image3d_config,
            )

            # update model
            self.encoder_image3d = ENCODER_FACTORY[self.config.encoder_image3d_type][1](self.config)
            self.connector_image3d = CONNECTOR_FACTORY[self.config.connector_image3d_type][1](self.config)
            self.connector_image3d.load_model(self.config.connector_image3d_path)

    def generate(self, input_ids, attention_mask, image, image3d, **generation_config):
        llm_inputs = self.prepare_inputs_for_multimodal(input_ids, attention_mask, image, image3d)
        if llm_inputs["inputs_embeds"] is None:
            llm_inputs["inputs_embeds"] = self.llm.get_input_embeddings()(llm_inputs["input_ids"])
        llm_inputs["use_cache"] = True
        llm_outputs = self.llm.generate(**llm_inputs, **generation_config)
        return llm_outputs

    def forward(self, input_ids, attention_mask, image, image3d, labels, **kwargs):  # peft包装完model后会额外加入无用参数，需要加上**kwargs
        llm_inputs = self.prepare_inputs_for_multimodal(input_ids, attention_mask, image, image3d, labels)
        if hasattr(self.config, "use_cache"):
            llm_inputs["use_cache"] = False
        llm_outputs = self.llm(**llm_inputs)
        return llm_outputs

    def prepare_inputs_for_multimodal(self, input_ids, attention_mask, image, image3d, labels=None):
        if image is None and image3d is None:
            return dict(
                input_ids=input_ids,
                inputs_embeds = None,
                attention_mask=attention_mask,
                labels=labels,
            )

        if image is not None:
            image = self.encoder_image(
                image,
                select_layer=self.config.encoder_image_select_layer,
                select_feature=self.config.encoder_image_select_feature,
            )
            image = self.connector_image(image)

        if image3d is not None:
            image3d = self.encoder_image3d(
                image3d,
                select_layer=self.config.encoder_image3d_select_layer,
                select_feature=self.config.encoder_image3d_select_feature,
            )
            image3d = self.connector_image3d(image3d)

        # 初始化labels
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX, device=self.device)

        # 根据attention_mask筛选input_ids、labels，转换为list格式
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        # 将image和image3d插入到input_ids、labels中，得到inputs_embeds、new_labels，支持多图操作
        cur_image_idx = 0
        cur_image3d_idx = 0
        inputs_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_image = (cur_input_ids == IMAGE_TOKEN_ID).sum()
            num_image3d = (cur_input_ids == IMAGE3D_TOKEN_ID).sum()
            if num_image == 0 and num_image3d == 0:
                inputs_embeds.append(self.llm.get_input_embeddings()(cur_input_ids))
                new_labels.append(labels[batch_idx])
                continue

            cur_labels = labels[batch_idx]
            image_token_idx_list = torch.where(cur_input_ids == IMAGE_TOKEN_ID)[0].tolist()
            image3d_token_idx_list = torch.where(cur_input_ids == IMAGE3D_TOKEN_ID)[0].tolist()
            special_token_idx_list = sorted(image_token_idx_list + image3d_token_idx_list+ [-1] + [cur_input_ids.size(0)])
            cur_input_ids_text = []
            cur_labels_text = []

            # 将cur_input_ids、cur_labels按照image和image3d的位置分割，得到cur_input_ids_text、cur_labels_text
            special_token_type = []
            for i in range(len(special_token_idx_list) - 1):
                start_idx = special_token_idx_list[i] + 1
                end_idx = special_token_idx_list[i + 1]
                cur_input_ids_text.append(cur_input_ids[start_idx:end_idx])
                cur_labels_text.append(cur_labels[start_idx:end_idx])

                if end_idx in image_token_idx_list:
                    special_token_type.append("image")
                elif end_idx in image3d_token_idx_list:
                    special_token_type.append("image3d")

            # 将cur_input_ids_text转换为cur_inputs_embeds_text，并按照image和image3d的位置切分
            split_lengths = [x.size(0) for x in cur_input_ids_text]
            cur_inputs_embeds_text = self.llm.get_input_embeddings()(torch.cat(cur_input_ids_text))
            cur_inputs_embeds_text = torch.split(cur_inputs_embeds_text, split_lengths)

            # 按照位置在cur_inputs_embeds_text中插入image和image3d，在cur_labels_text中插入IGNORE_INDEX，得到cur_inputs_embeds、cur_new_labels
            cur_inputs_embeds = []
            cur_new_labels = []
            for i in range(num_image + num_image3d + 1):
                cur_inputs_embeds.append(cur_inputs_embeds_text[i])
                cur_new_labels.append(cur_labels_text[i])

                if i < num_image + num_image3d:
                    if special_token_type[i] == "image":
                        cur_inputs_embeds.append(image[cur_image_idx])
                        cur_new_labels.append(
                            torch.full(
                                (image[cur_image_idx].size(0),),
                                IGNORE_INDEX,
                                device=self.device,
                            )
                        )
                        cur_image_idx += 1
                    elif special_token_type[i] == "image3d":
                        cur_inputs_embeds.append(image3d[cur_image3d_idx])
                        cur_new_labels.append(
                            torch.full(
                                (image3d[cur_image3d_idx].size(0),),
                                IGNORE_INDEX,
                                device=self.device,
                            )
                        )
                        cur_image3d_idx += 1

            # 将cur_inputs_embeds、cur_new_labels合并到inputs_embeds、new_labels
            inputs_embeds.append(torch.cat(cur_inputs_embeds))
            new_labels.append(torch.cat(cur_new_labels))

        # 将inputs_embeds、new_labels按照llm_max_length截断
        inputs_embeds = [x[: self.config.llm_max_length] for x in inputs_embeds]
        labels = [x[: self.config.llm_max_length] for x in new_labels]

        # 初始化inputs_embeds_padded、labels_padded、attention_mask_padded、position_ids_padded
        batch_size = len(inputs_embeds)
        max_len = max(x.size(0) for x in inputs_embeds)
        inputs_embeds_padded = []
        labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            device=self.device,
        )
        attention_mask_padded = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
            device=self.device,
        )

        # 按照llm_padding_side填充inputs_embeds_padded、labels_padded、attention_mask_padded、position_ids_padded
        for i, (cur_inputs_embeds, cur_labels) in enumerate(zip(inputs_embeds, labels)):
            cur_len = cur_inputs_embeds.size(0)
            zero_inputs_embeds = torch.zeros(
                (max_len - cur_len, cur_inputs_embeds.size(1)),
                dtype=self.dtype,
                device=self.device,
            )
            if self.config.llm_padding_side == "left":
                inputs_embeds_padded.append(torch.cat((zero_inputs_embeds, cur_inputs_embeds)))
                labels_padded[i, -cur_len:] = cur_labels
                attention_mask_padded[i, -cur_len:] = True
            else:
                inputs_embeds_padded.append(torch.cat((cur_inputs_embeds, zero_inputs_embeds)))
                labels_padded[i, :cur_len] = cur_labels
                attention_mask_padded[i, :cur_len] = True
        inputs_embeds_padded = torch.stack(inputs_embeds_padded)

        return dict(
            input_ids=None,
            inputs_embeds=inputs_embeds_padded,
            attention_mask=attention_mask_padded,
            labels=labels_padded,
        )
