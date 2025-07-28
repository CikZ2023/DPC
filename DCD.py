import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from text_attention import llama_modify

import torch.nn.functional as F
from constants import INSTRUCTION_TEMPLATE, SYSTEM_MESSAGE
from eval_data_loader import COCODataSet
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessorList
from torch.utils.data import Subset
import threading
import torch
import torch.nn.functional as F
from  utils import  extract_attention_weights
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from global_var import global_attention_weights
from tqdm import tqdm
from utils import set_act_get_hooks, remove_hooks

count = 0
class dcd(LogitsProcessor):
    def __init__(
            self,
            kwargs,
            guidance_scale,
            model,
            prompt_tokens,
            input_type="inputs_ids",
            start_layer=0,
            end_layer=32,
            use_attn=True,
            alpha=0.4,
            b=0.2,
            use_cfg=True,
            image=None,
            model_loader = None


    ):
        self.kwargs=kwargs
        self.guidance_scale = guidance_scale
        self.model = model
        self.out = None
        self.input_type = input_type
        self.prompt_tokens = prompt_tokens
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.use_attn = use_attn
        self.alpha = alpha
        self.b = b
        self.use_cfg = use_cfg
        self.image=kwargs["images"],
        self.attention_mask = None
        self.output1 = None
        self.output2 = None
        self.lock = threading.Lock()
        self.token_count = 0
        self.model_loader = model_loader
        self.insignificant_layers = [7, 11, 15, 16, 18, 19, 20, 23, 32]
        self.negative_layers = [14, 22, 24]
        self.positive_layers = [i for i in range(2, 33) if
                                i not in self.insignificant_layers and i not in self.negative_layers]
        self.T=0.2
        self.base_array = np.load('./llava_array.npy')

    def get_current_attentions(self):
        """获取当前所有层的注意力权重"""
        attentions = {}

        # 从activations_中获取
        if hasattr(self.model.model, 'activations_'):
            for layer_idx in range(0, self.end_layer):
                key = f"attn_out_{layer_idx}"
                if key in self.model.model.activations_:
                    attn_list = self.model.model.activations_[key]
                    if attn_list:
                        # 获取最新的注意力权重
                        attentions[layer_idx] = attn_list[-1]

        return attentions

    def calculate_attention_ratios(self, v_attnw_matrix, t_attnw_matrix):
        num_layers, num_heads = v_attnw_matrix.shape

        layer_wise_ratios = []
        global_ratio = 0

        for l in range(num_layers):
            layer_ratio = 0
            for h in range(num_heads):
                visual_value = v_attnw_matrix[l, h]
                text_value = t_attnw_matrix[l, h]
                ratio = visual_value / text_value if text_value != 0 else 0
                layer_ratio += ratio

            avg_layer_ratio = layer_ratio / num_heads
            layer_wise_ratios.append(avg_layer_ratio)
            global_ratio += avg_layer_ratio

        # global_ratio /= num_layers

        return layer_wise_ratios

    def adjust_layer_parameters(self, layer_wise_ratios):
        for i in range(self.start_layer, self.end_layer):
            layer_index = i - self.start_layer  # 调整索引以匹配 layer_wise_ratios

            x = layer_wise_ratios[layer_index]
            y = self.base_array[layer_index]

            if x < y:
                delta = self.T * (abs(x - y) / y)
                self.model.model.layers[i].self_attn.alpha = min(0.6, self.alpha + delta)
                self.model.model.layers[i].self_attn.b = min(0.6, self.b + delta)
            else:  # 不明显的层
                self.model.model.layers[i].self_attn.alpha = self.alpha
                self.model.model.layers[i].self_attn.b = self.b
    def clear_attention_caches(self):
        # 清除所有存储的注意力权重和缓存
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, 'has_saved_original_attn_weights'):
                layer.self_attn.has_saved_original_attn_weights = False
            if hasattr(layer.self_attn, 'original_attention_weights'):
                layer.self_attn.original_attention_weights = None
            if hasattr(layer.self_attn, 'attention_cache'):
                layer.self_attn.attention_cache = []

        # 如果模型有 activations_ 字典，也要清除
        if hasattr(self.model.model, 'activations_'):
            # 仅清除注意力输出，而不是所有激活
            keys_to_clear = [k for k in self.model.model.activations_ if k.startswith('attn_out_')]
            for k in keys_to_clear:
                self.model.model.activations_[k] = []

    def __call__(self, input_ids, scores):
        self.token_count += 1

        scores = F.log_softmax(scores, dim=-1)

        if self.guidance_scale == 1:
            return scores

        # 获取当前的注意力权重
        current_attentions = self.get_current_attentions()
        if current_attentions:
            # 方法1：将所有层的注意力权重堆叠成一个大张量
            attention_list = []
            layer_indices = []

            for layer_idx in sorted(current_attentions.keys()):
                attn = current_attentions[layer_idx]
                attention_list.append(attn)
                layer_indices.append(layer_idx)

            if attention_list:
                # 堆叠所有层的注意力权重
                # 结果形状: [num_layers, batch_size, num_heads, seq_len, seq_len]
                stacked_attentions = torch.stack(attention_list, dim=0)
            v_attnw_matrix, t_attnw_matrix, attnw_matrix = extract_attention_weights(stacked_attentions,self.model_loader)

        # 清楚缓存
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, 'has_saved_original_attn_weights'):
                layer.self_attn.has_saved_original_attn_weights = False
            if hasattr(layer.self_attn, 'attention_cache'):
                layer.self_attn.attention_cache = []
        layer_wise_ratios = self.calculate_attention_ratios(v_attnw_matrix.cpu().numpy(),t_attnw_matrix.cpu().numpy())

        for i in range(self.start_layer, self.end_layer):
            self.model.model.layers[i].self_attn.use_cfg = True
            self.model.model.layers[i].self_attn.use_attn = True
            self.adjust_layer_parameters(layer_wise_ratios)

        if self.output1 is None:
            if self.input_type == "inputs_ids":
                self.output1 = self.model(input_ids=input_ids, images=self.image[0], use_cache=True)
            elif self.input_type == "inputs_embeds":
                self.output1 = self.model(inputs_embeds=self.prompt_tokens, use_cache=True)
            else:
                print("Neither input_ids nor inputs_embeds is provided.")
        else:
            with torch.no_grad():
                self.output1 = self.model(
                    input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=self.output1.past_key_values,
                )

        image_focus_logits = F.log_softmax(self.output1.logits[:, -1, :], dim=-1)



        for i in range(self.start_layer, self.end_layer):
            self.model.model.layers[i].self_attn.use_cfg = False
            self.model.model.layers[i].self_attn.use_attn = True
            # self.adjust_layer_parameters(layer_wise_ratios)
        if self.output2 is None:
            if self.input_type == "inputs_ids":
                self.output2 = self.model(input_ids=self.prompt_tokens, images=self.image[0], use_cache=True)
            elif self.input_type == "inputs_embeds":
                self.output2 = self.model(inputs_embeds=self.prompt_tokens, use_cache=True)
            else:
                print("Neither input_ids nor inputs_embeds is provided.")
        else:
            with torch.no_grad():
                self.output2 = self.model(
                    input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=self.output2.past_key_values,
                )
        text_focus_logits = F.log_softmax(self.output2.logits[:, -1, :], dim=-1)

        for i in range(self.start_layer, self.end_layer):
            self.model.model.layers[i].self_attn.use_cfg = False
            self.model.model.layers[i].self_attn.use_attn = False
            self.model.model.layers[i].self_attn.alpha = self.alpha
            self.model.model.layers[i].self_attn.b = self.b

        self.clear_attention_caches()

        cutoff = torch.log(torch.tensor(0.1)) + scores.max(dim=-1, keepdim=True).values


        out = (
                self.guidance_scale * (image_focus_logits - text_focus_logits) + text_focus_logits
        )

        cd_logits =out.masked_fill(scores < cutoff, -float("inf"))


        return cd_logits






