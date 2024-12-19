import torch
from peft import LoraConfig, get_peft_model


class BaseTrainingRecipe:
    def __init__(self, training_arguments):
        self.training_arguments = training_arguments

    def __call__(self, model):
        model = self.training_model_converse(model)
        model = self.tune_type_setting(model)

        # `use_cache=True` is incompatible with gradient checkpointing.
        if self.training_arguments.gradient_checkpointing:
            model.config.use_cache = False

        return model

    def training_model_converse(self, model):
        if self.training_arguments.tune_type_llm == "lora":
            lora_config = LoraConfig(
                r=self.training_arguments.llm_lora_r,
                lora_alpha=self.training_arguments.llm_lora_alpha,
                lora_dropout=self.training_arguments.llm_lora_dropout,
                bias=self.training_arguments.llm_lora_bias,
                task_type="CAUSAL_LM",
            )
            model.llm = get_peft_model(model.llm, lora_config)

        return model

    def tune_type_setting(self, model):
        model = self._llm_tune_type_setting(model)
        if model.encoder_image is not None:
            model = self._encoder_image_tune_type_setting(model)
            model = self._connector_image_tune_type_setting(model)
        if model.encoder_image3d is not None:
            model = self._encoder_image3d_tune_type_setting(model)
            model = self._connector_image3d_tune_type_setting(model)
        return model

    def _llm_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_llm.lower()
        if tune_type == "full":
            model.llm.requires_grad_(True)
        elif tune_type == "frozen":
            model.llm.requires_grad_(False)

        # gradient checkpointing是一种优化训练深度神经网络时内存占用的技术
        self.support_gradient_checkpoint(model.llm, self.training_arguments.gradient_checkpointing)

        return model

    def _encoder_image_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_encoder_image.lower()
        if tune_type == "full":
            model.encoder_image.requires_grad_(True)
        elif tune_type == "frozen":
            model.encoder_image.requires_grad_(False)
        return model

    def _connector_image_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_connector_image.lower()
        if tune_type == "full":
            for p in model.connector_image.parameters():
                p.requires_grad = True
        elif tune_type == "frozen":
            for p in model.connector_image.parameters():
                p.requires_grad = False
        return model

    def _encoder_image3d_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_encoder_image3d.lower()
        if tune_type == "full":
            model.encoder_image3d.requires_grad_(True)
        elif tune_type == "frozen":
            model.encoder_image3d.requires_grad_(False)
        return model

    def _connector_image3d_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_connector_image3d.lower()
        if tune_type == "full":
            for p in model.connector_image3d.parameters():
                p.requires_grad = True
        elif tune_type == "frozen":
            for p in model.connector_image3d.parameters():
                p.requires_grad = False
        return model

    def support_gradient_checkpoint(self, model, gradient_checkpointing=False):
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        if gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def save(self, model, trainer):
        if trainer.deepspeed:
            torch.cuda.synchronize()

        # if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if self.training_arguments.tune_type_llm == "lora":
            model.llm = model.llm.merge_and_unload()
        trainer.save_model(self.training_arguments.output_dir)
