import cv2
import random
import inflect
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from typing import List

engine = inflect.engine()


def setup_seeds():
    seed = 927

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def disable_torch_init():
    """
    Disable redundant torch default initialization to accelerate model creation.
    Copied from llava.utils
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def set_act_get_hooks1(model, attn_out=False):
    """
    Set hooks to capture attention weights (with softmax).
    """
    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})
        else:
            model.activations_ = {}

    def get_attention_weights_hook(name, layer_idx):
        def hook(module, input, output):
            # Get original attention weights (before softmax)
            if hasattr(module, 'original_attention_weights'):
                attn_weights = module.original_attention_weights.clone()

                # Apply softmax to get final attention weights
                attn_weights_softmax = torch.nn.functional.softmax(
                    attn_weights,
                    dim=-1,
                    dtype=torch.float32
                ).to(attn_weights.dtype)

                if name not in model.activations_:
                    model.activations_[name] = [attn_weights_softmax.detach()]
                else:
                    model.activations_[name].append(attn_weights_softmax.detach())

                # Save raw weights (optional)
                raw_name = f"attn_raw_{layer_idx}"
                if raw_name not in model.activations_:
                    model.activations_[raw_name] = [attn_weights.detach()]
                else:
                    model.activations_[raw_name].append(attn_weights.detach())

        return hook

    hooks = []
    for i in range(model.config.num_hidden_layers):
        if attn_out:
            hook = model.layers[i].self_attn.register_forward_hook(
                get_attention_weights_hook(f"attn_out_{i}", i)
            )
            hooks.append(hook)

    return hooks

def find_text_position(decoded_tokens, target_text):
    """
    Find the position of target text in decoded tokens list.
    :param decoded_tokens: List of decoded tokens
    :param target_text: Target text to find
    :return: Position index of the target text
    """
    positions = 0
    target_text = target_text.lower()

    for i, token in enumerate(decoded_tokens):
        if target_text in token.lower():
            positions = i

    return positions
def extract_attention_weights(attn_weights, model_loader):
    """
    Extract and process attention weights from model outputs.

    Params:
    -------
    attn_weights: torch.Tensor
        Raw attention weights from model
    model_loader: object
        Contains token position indices

    Return:
    -------
    vision_attn_matrix: torch.Tensor
        Attention weights over vision tokens
    text_attn_matrix: torch.Tensor
        Attention weights over text tokens
    original_weights: torch.Tensor
        Original attention weights
    """
    if attn_weights.dim() == 5:
        head_key_weights = attn_weights[:, 0, :, 0, :]
    elif attn_weights.dim() == 4:
        head_key_weights = attn_weights[:, :, 0, :]
    else:
        raise ValueError(f"Unsupported dimension: {attn_weights.dim()}")

    num_layers, num_heads, key_len = head_key_weights.shape
    device = attn_weights.device
    dtype = attn_weights.dtype

    vision_attn_matrix = torch.zeros((num_layers, num_heads), device=device)
    text_attn_matrix = torch.zeros((num_layers, num_heads), device=device)
    original_weights = torch.zeros((num_layers, num_heads, key_len), device=device, dtype=dtype)

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            weights = head_key_weights[layer_idx, head_idx]

            if weights.dim() > 1:
                weights = weights.flatten()[:key_len]

            original_weights[layer_idx, head_idx] = weights.clone()

            vision_start = model_loader.img_start_idx
            vision_end = model_loader.img_end_idx
            vision_attn = weights[vision_start:vision_end].sum().item()
            vision_attn_matrix[layer_idx, head_idx] = vision_attn

            text_before = weights[:model_loader.text_end_idx_before_img].sum().item()
            text_after = weights[model_loader.text_start_idx_after_img:model_loader.text_end_idx].sum().item()
            text_attn_matrix[layer_idx, head_idx] = text_before + text_after

    return vision_attn_matrix, text_attn_matrix, original_weights
