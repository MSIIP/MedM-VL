# Parameter Interpretation

## 1. Training Recipe
+ `common`

## 2. LLM and Template
| llm_name_or_path                     | llm_type | conv_version (finetune) |
| ------------------------------------ | -------- | ----------------------- |
| `microsoft/phi-2`                    | `phi`    | `phi`                   |
| `microsoft/Phi-3-mini-128k-instruct` | `phi3`   | `phi3`                  |
| `meta-llama/Llama-3.2-3B-Instruct`   | `llama3` | `llama3`                |
| `Qwen/Qwen2.5-3B-Instruct`           | `qwen2`  | `qwen2`                 |

+ tune_type_llm: `frozen`, `lora`, `full`

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
| `shiym2000/MedM-CLIP-CT`     | `medmclip`           |

+ tune_type_encoder_image3d: `frozen`, `full`

| connector_image3d_name | connector_image3d_type |
| ---------------------- | ---------------------- |
| `mlp2x_gelu`           | `mlp`                  |
| `mlp2x_gelu`           | `spatial_pooling`      |
| `mlp2x_gelu`           | `attn_pooling`         |

+ tune_type_connector_image3d: `frozen`, `full`