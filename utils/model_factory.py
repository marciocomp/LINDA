# @author: Marcio Lopes
import gc

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from utils.linda_logger import logger

from utils.models.gemma_wrapper import GemmaShardWrapper
from utils.models.resnet50 import ResNet50Split


class ModelFactory:
    """
    Universal factory for instantiating model shards.
    """

    @staticmethod
    def get_model_shard(model_name,
                        model_type,
                        start_layer,
                        end_layer,
                        is_first=False,
                        is_last=False,
                        device=None,
                        hf_token=None,
                        num_classes=10):


        logger.info(f"Factory Building Shard: {model_name} | Layers: {start_layer}-{end_layer} | Device: {device}")

        if model_type == 'vision':
            return ModelFactory._get_vision_shard(
                model_name=model_name,
                start_layer=start_layer,
                end_layer=end_layer,
                num_classes=num_classes,
                device=device
            )
        elif model_type == 'llm':
            return ModelFactory._get_llm_shard(
                model_name=model_name,
                start_layer=start_layer,
                end_layer=end_layer,
                is_first=is_first,
                is_last=is_last,
                hf_token=hf_token,
                device=device
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    @staticmethod
    def _get_vision_shard(model_name, start_layer, end_layer, num_classes, device):
        """
        Slices ResNet using logical block definitions.
        """
        if "resnet50" in model_name:
            # pretrained=True for transfer learning; ideally from config
            full_model_wrapper = ResNet50Split(num_classes=num_classes, pretrained=True)

            all_blocks = full_model_wrapper.layers
            total_blocks = len(all_blocks)

            if start_layer >= total_blocks:
                raise ValueError(f"Start layer {start_layer} out of bounds (Total Blocks: {total_blocks})")

            if end_layer is None or end_layer == -1 or end_layer > total_blocks:
                shard_layers = all_blocks[start_layer:]
            else:
                shard_layers = all_blocks[start_layer:end_layer]

            shard_model = nn.Sequential(*shard_layers)
            return shard_model.to(device)

        else:
            raise NotImplementedError(f"Vision model {model_name} not supported yet.")

    @staticmethod
    def _get_llm_shard(model_name, start_layer, end_layer, is_first, is_last, hf_token, device):

        if "gemma" in model_name:
            if not hf_token: raise ValueError("HF_TOKEN required")
            full_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                              token=hf_token,
                                                              low_cpu_mem_usage=True,
                                                              torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32 ,
                                                              # revision="float16",
                                                              )

            if device.type == "cuda":
                logger.info(f"Setting checkpointing to GPU")
                full_model.gradient_checkpointing_enable()
            else:
                logger.info(f"Checkpointing was not enabled to CPU")


            total_layers = len(full_model.model.layers)
            if end_layer is None or end_layer > total_layers: end_layer = total_layers

            shard = GemmaShardWrapper(full_model, start_layer, end_layer, is_first, is_last)

            del full_model
            gc.collect()

            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()

            return shard.to(device)
        else:
            raise NotImplementedError(f"LLM {model_name} not supported")
