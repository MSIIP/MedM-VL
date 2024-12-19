# Parameter Interpretation

## 1. Training Recipe
+ `common`

## 2. LLM and Template
| llm_name_or_path                     | llm_type | llm_padding_side | conv_version                       |
| ------------------------------------ | -------- | ---------------- | ---------------------------------- |
| `microsoft/phi-2`                    | `phi`    | `right`          | `pretrain/pretrain_image3d/phi`    |
| `microsoft/Phi-3-mini-128k-instruct` | `phi3`   | `right`          | `pretrain/pretrain_image3d/phi3`   |
| `meta-llama/Llama-3.2-3B-Instruct`   | `llama3` | `right`          | `pretrain/pretrain_image3d/llama3` |
| `Qwen/Qwen2.5-3B-Instruct`           | `qwen2`  | `right`          | `pretrain/pretrain_image3d/qwen2`  |

+ tune_type_llm: `frozen`, `full`, `lora`

## 3. Encoder and Connector (image)
| encoder_image_name_or_path          | encoder_image_type |
| ----------------------------------- | ------------------ |
| `openai/clip-vit-large-patch14-336` | `clip`             |
| `google/siglip-so400m-patch14-384`  | `siglip`           |

+ tune_type_encoder_image: `frozen`, `full`

| connector_image_name | connector_image_type |
| -------------------- | -------------------- |
| `mlp2x_gelu`         | `mlp`                |

+ tune_type_connector_image: `frozen`, `full`

## 4. Encoder and Connector (image3d)
| encoder_image3d_name_or_path | encoder_image3d_type |
| ---------------------------- | -------------------- |
| `GoodBaiBai88/M3D-CLIP`      | `m3dclip`            |

+ tune_type_encoder_image3d: `frozen`, `full`

| connector_image3d_name | connector_image3d_type |
| ---------------------- | ---------------------- |
| `mlp2x_gelu`           | `mlp`                  |
| `mlp2x_gelu`           | `spatial_pooling`      |

+ tune_type_connector_image3d: `frozen`, `full`