# @author: Marcio Lopes
import torch
import torch.nn as nn
import gc
from typing import Optional, Union, Tuple, List


class GemmaShardWrapper(nn.Module):
    """
    Optimized wrapper for Gemma model sharding.
    Compatible with Python 3.8+ and Transformers 4.46.3.

    - Destructive pruning to free RAM.
    - Automatic dtype conversion (Float16/Float32).
    - sqrt(dim) scale correction.
    - RoPE abstraction (managed by transformers).
    """

    def __init__(self, original_model, start_layer: int, end_layer: int, is_first: bool = False, is_last: bool = False):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        self.config = original_model.config

        # --- SLICING & PRUNING (Destructive) ---
        layers_list = original_model.model.layers
        total_layers = len(layers_list)

        if end_layer > total_layers:
            end_layer = total_layers

        sliced_layers = layers_list[start_layer:end_layer]

        # Replace the original list with the slice so GC can free the rest
        original_model.model.layers = nn.ModuleList(sliced_layers)

        # --- REMOVE UNUSED COMPONENTS ---
        if not is_first:
            if hasattr(original_model.model, 'embed_tokens'):
                del original_model.model.embed_tokens
                original_model.model.embed_tokens = None

        if not is_last:
            if hasattr(original_model.model, 'norm'):
                del original_model.model.norm
                original_model.model.norm = nn.Identity()

            if hasattr(original_model, 'lm_head'):
                del original_model.lm_head
                original_model.lm_head = None
        else:
            self.lm_head = original_model.lm_head

        self.internal_model = original_model.model


    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **kwargs
    ):
        """
        Args:
            input_ids: LongTensor (tokens) if is_first=True,
                       FloatTensor (hidden states) if is_first=False.
        """

        if self.is_first:
            outputs = self.internal_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )
        else:
            # TIER 2/3: Input is hidden states from the previous node
            try:
                target_dtype = next(self.internal_model.parameters()).dtype
                if input_ids.dtype != target_dtype:
                    input_ids = input_ids.to(target_dtype)
            except StopIteration:
                pass

            # Gemma applies input * sqrt(dim) at embedding; undo the scaling
            # since the data arrived already scaled from the previous node
            scale_factor = (self.config.hidden_size ** 0.5)
            x_scaled = input_ids / scale_factor

            outputs = self.internal_model(
                inputs_embeds=x_scaled,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True
            )

        hidden_states = outputs.last_hidden_state

        if self.is_last:
            logits = self.lm_head(hidden_states)
            return logits

        return hidden_states