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
    Disable the redundant torch default initialization to accelerate model creation.
    Copied from llava.utils
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def set_act_get_hooks1(model, attn_out=False):
    # 确保activations_属性存在且初始化
    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})
        else:
            model.activations_ = {}

    def get_attention_weights_hook(name, layer_idx):
        def hook(module, input, output):
            # 获取保存的原始注意力权重（在softmax之前）
            if hasattr(module, 'original_attention_weights'):
                # 如果需要softmax后的权重
                attn_weights = module.original_attention_weights.clone()

                # 应用softmax获得最终的注意力权重
                attn_weights_softmax = torch.nn.functional.softmax(
                    attn_weights,
                    dim=-1,
                    dtype=torch.float32
                ).to(attn_weights.dtype)

                if name not in model.activations_:
                    model.activations_[name] = [attn_weights_softmax.detach()]
                else:
                    model.activations_[name].append(attn_weights_softmax.detach())

                # 同时保存原始权重（可选）
                raw_name = f"attn_raw_{layer_idx}"
                if raw_name not in model.activations_:
                    model.activations_[raw_name] = [attn_weights.detach()]
                else:
                    model.activations_[raw_name].append(attn_weights.detach())

        return hook

    # 注册hook到指定层
    hooks = []
    for i in range(model.config.num_hidden_layers):
        if attn_out:
            hook = model.layers[i].self_attn.register_forward_hook(
                get_attention_weights_hook(f"attn_out_{i}", i)
            )
            hooks.append(hook)

    return hooks
def set_act_get_hooks(model, attn_out=False):

    '''
    Set hooks to capture activations.
    '''
    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})
        else:
            model.activations_ = {}

    def get_activation(name):
        def hook(module, input, output):
            from global_var import global_attention_weights,lock


            if "attn_out" in name:
                if name not in model.activations_:
                    model.activations_[name] = [output[0].squeeze(0).detach()]
                else:
                    model.activations_[name].append(output[0].squeeze(0).detach())

        return hook

    hooks = []
    for i in range(model.config.num_hidden_layers):
        if attn_out:
            hooks.append(model.layers[i].self_attn.register_forward_hook(get_activation(f"attn_out_{i}")))

    return hooks


# Always remove your hooks, otherwise things will get messy.
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def get_only_attn_out_contribution(
        model, tokenizer, outputs, text: str, output_start_idx: int
):
    ''' Get the Attn. Sublayer contribution of the selected object token to the final prediction.
    Params:
    -------
    model: the language component of the LVLMs
    tokenizer: the tokenizer of the model
    outputs: dict
        the outputs of the model containing the 'output sequence' and 'scores' attributes
    text: str
        real or hallucinated text
    output_start_idx: int (= input length - 1)

    Return:
    -------
    records_attn: list
        the attention contribution over layers
    '''
    selected_token_id = tokenizer(text, add_special_tokens=False)["input_ids"][0]
    # the first index is adoptted if there are multiple occurrences
    token_in_generation_idx = torch.nonzero(outputs['sequences'][0][1:] == selected_token_id)[0].item()
    final_probs = F.softmax(outputs['scores'][token_in_generation_idx], dim=-1)
    _, topk_token_ids = final_probs.topk(1)
    topk_token_ids = topk_token_ids[0]

    linear_projector = model.lm_head # llava1.5

    records_attn = []
    for layer_i in range(model.model.config.num_hidden_layers):
        # ATTN
        attn_out = (
            model.model.activations_[f"attn_out_{layer_i}"][output_start_idx + token_in_generation_idx, :]
            ).clone().detach()
        proj = linear_projector(attn_out)
        attn_logit = proj.cpu().detach().numpy()
        records_attn.append(attn_logit[topk_token_ids])

    return records_attn

def find_text_position(decoded_tokens, target_text):
    """
    Find the position of the target text in the decoded tokens list.
    :param decoded_tokens: List of decoded tokens.
    :param target_text: Target text to find.
    :return: List of positions where the target text is found.
    """
    positions = 0
    target_text = target_text.lower()  # 转换为小写以进行大小写不敏感的匹配

    for i, token in enumerate(decoded_tokens):
        # 检查当前 token 是否包含目标文本
        if target_text in token.lower():
            positions = i

    return positions


def is_meaningful_token(token_str):
    """判断 token 是否有实际意义"""
    # 标记有实际含义的类型
    meaningful_prefixes = ['__', '_']
    meaningful_symbols = ['.', ',', ':', '!', '?', "'", '"', '[UNK]']

    # 1. 检查是否有意义的前缀
    if any(token_str.startswith(prefix) for prefix in meaningful_prefixes):
        return True

    # 2. 检查是否是特定符号（可能有含义）
    if token_str in meaningful_symbols:
        return True

    # 3. 排除完全无意义的标记
    if token_str in ['<s>', '</s>', '<0x0A>', 'Ġ']:  # 结束标记、换行符等
        return False

    # 4. 检查是否是普通字母单词
    if len(token_str) > 1 and token_str[0].isalpha():
        return True

    return False
def attnw_over_vision_and_text_layer_head_selected_text(
        text: str, outputs, tokenizer, vision_token_start, vision_token_end,
        text_start_idx, text_end_idx_before_img, text_start_idx_after_img, text_end_idx,
        sort_heads=False
):
    ''' Get the attention weights over the image tokens and text tokens for the selected object text.
    Params:
    -------
    text: str
        the selected object text
    outputs: dict
        the outputs of the model containing the 'attentions' and 'sequences' attributes
    tokenizer: the tokenizer of the model
    vision_token_start/_end: int
        the start/end index of the image tokens
    text_start_idx, text_end_idx_before_img, text_start_idx_after_img, text_end_idx: int
        the start/end indices of the text tokens before and after the image

    Return:
    -------
    vision_attnw_matrix: np.ndarray (num_layers, num_heads)
        the attention weights over the image tokens for the selected object text over layers and heads
    text_attnw_matrix: np.ndarray (num_layers, num_heads)
        the attention weights over the text tokens for the selected object text over layers and heads
    original_attnw_matrix: np.ndarray (num_layers, num_heads, sequence_length)
        the original attention weights for the selected object text over layers and heads
    token_in_generation_idx: int
        the index of the selected token in the generated sequence
    '''

    decoded_tokens =[]
    for token_id in outputs['sequences'][0]:

        if (token_id.item() == -200):
            decoded_tokens.append("[UNK]")

        else:
            decoded_tokens.append(tokenizer.convert_ids_to_tokens(token_id.item()))




    token_in_generation_idx = find_text_position(decoded_tokens, text)-50





    text_attnw_layers_heads = outputs['attentions'][token_in_generation_idx]
    num_layers = len(text_attnw_layers_heads)
    num_heads = text_attnw_layers_heads[0].shape[1]
    sequence_length = text_attnw_layers_heads[0].shape[-1]

    vision_attnw_matrix = torch.zeros((num_layers, num_heads, vision_token_end - vision_token_start))
    text_attnw_matrix = torch.zeros((num_layers, num_heads, text_end_idx - text_start_idx_after_img))
    prompt_attnw_matrix = torch.zeros((num_layers, num_heads, text_end_idx_before_img - text_start_idx))
    original_attnw_matrix = torch.zeros((num_layers, num_heads, sequence_length))

    for i, layer_attnw in enumerate(text_attnw_layers_heads):
        for j, head_attnw in enumerate(layer_attnw.squeeze(0)):
            vision_attnw_matrix[num_layers - 1 - i, j] = head_attnw[-1][vision_token_start:vision_token_end]

            # Extract attention weights for text tokens before and after the image
            text_attnw_matrix[num_layers - 1 - i, j] = head_attnw[-1][text_start_idx_after_img:text_end_idx]
            prompt_attnw_matrix[num_layers - 1 - i, j] = head_attnw[-1][text_start_idx:text_end_idx_before_img]

            # Store the original attention weights
            original_attnw_matrix[i, j] = head_attnw[-1]

    if sort_heads:
        vision_attnw_matrix, _ = torch.sort(vision_attnw_matrix, dim=1, descending=True)
        text_attnw_matrix, _ = torch.sort(text_attnw_matrix, dim=1, descending=True)

    vision_attnw_matrix = vision_attnw_matrix.numpy()
    text_attnw_matrix = text_attnw_matrix.numpy()
    original_attnw_matrix = original_attnw_matrix.numpy()
    prompt_attnw_matrix = prompt_attnw_matrix.numpy()

    return vision_attnw_matrix, text_attnw_matrix, prompt_attnw_matrix, original_attnw_matrix

def extract_attention_weights(attn_weights, model_loader):
    # 输入维度: [num_layers=30, batch=1, num_heads=32, query_len=1, key_len=625~629]

    if attn_weights.dim() == 5:
        # 标准5维输入 [layers, batch, heads, query, key]
        head_key_weights = attn_weights[:, 0, :, 0, :]  # 显式选择batch=0和query=0
    elif attn_weights.dim() == 4:
        # 处理可能的4维输入 [layers, heads, query, key]
        head_key_weights = attn_weights[:, :, 0, :]  # 压缩query维度
    else:
        raise ValueError(f"不支持的维度: {attn_weights.dim()}")

   # 应为 [layers, heads, key_len]

    # 2. 初始化输出张量（保留原始数据类型）
    num_layers, num_heads, key_len = head_key_weights.shape
    device = attn_weights.device
    dtype = attn_weights.dtype

    vision_attn_matrix = torch.zeros((num_layers, num_heads), device=device)
    text_attn_matrix = torch.zeros((num_layers, num_heads), device=device)
    original_weights = torch.zeros((num_layers, num_heads, key_len), device=device, dtype=dtype)

    # 3. 安全索引赋值（强制一维化）
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            weights = head_key_weights[layer_idx, head_idx]

            # 关键修复：展平为1D张量
            if weights.dim() > 1:
                weights = weights.flatten()[:key_len]  # 取前key_len个元素

            original_weights[layer_idx, head_idx] = weights.clone()

            # 后续计算逻辑保持不变...
            vision_start = model_loader.img_start_idx
            vision_end = model_loader.img_end_idx
            vision_attn = weights[vision_start:vision_end].sum().item()
            vision_attn_matrix[layer_idx, head_idx] = vision_attn

            text_before = weights[:model_loader.text_end_idx_before_img].sum().item()
            text_after = weights[model_loader.text_start_idx_after_img:model_loader.text_end_idx].sum().item()
            text_attn_matrix[layer_idx, head_idx] = text_before + text_after

    # 按头排序（可选）

    return vision_attn_matrix, text_attn_matrix, original_weights
def logitLens_of_vision_tokens(
        model, tokenizer, input_ids, outputs, token_range: List[int], layer_range: List[int],
        logits_warper, logits_processor
):
    ''' Retrieve the text token in the vocabulary with the highest prob
        for each image token in selected token index range.
    Params:
    -------
    model: the language component of the LVLMs
    tokenizer: the tokenizer of the model
    input_ids: tensor
        the input sequence of the model
    outputs: dict
        the outputs of the model containing the 'hidden_states' attribute
    token_range: list
        [start_image_token_idx, end_image_token_idx]
    layer_range: list
        the range of layers to be considered

    Returns:
    --------
    layer_max_prob: tensor (len(layer_range), image_token_num)
        the max prob predicted by the linear projector before softmax on the hidden states of each image token
        over the selected layers
    layer_words: list of list
        the retrieved text token for each image token over the selected layers
    '''
    layer_max_prob = torch.zeros((1, token_range[1] - token_range[0]))
    layer_words = []
    for i in layer_range:
        hidden_state = outputs['hidden_states'][0][i + 1].squeeze(0)
        hidden_state = hidden_state[token_range[0]:token_range[1]].clone().detach()
        logits = model.lm_head(hidden_state).cpu().float() # llava1.5
        logits = F.log_softmax(logits, dim=-1)
        logits_processed = logits_processor(input_ids, logits)
        logits = logits_warper(input_ids, logits_processed)

        probs = F.softmax(logits, dim=-1)
        vals, ids = probs.max(dim=-1)
        layer_max_prob = torch.cat([vals.unsqueeze(0).cpu().detach(), layer_max_prob], dim=0)
        layer_words.append([tokenizer.decode(id, skip_special_tokens=True) for id in ids])

    layer_max_prob = layer_max_prob[:-1] # drop the all zero row

    return layer_max_prob, layer_words


def logitLens_of_vision_tokens_with_discrete_range(
        model, tokenizer, input_ids, outputs, vision_token_start: int,
        discrete_range: List[List[int]], layer_range: List[int],
        logits_warper, logits_processor, fig_name: str = None
):
    '''
    Refer to the function `logitLens_of_vision_tokens` for the detailed description.
    '''
    assert(hasattr(outputs, 'hidden_states'))

    vision_discrete_range = [
        [vision_token_start + range_i[0], vision_token_start + range_i[1] + 1] for range_i in discrete_range
    ]

    each_range_layer_prob_list = []
    each_range_layer_words_list = []
    x_ticks = []
    y_ticks = [f'{i} h_out' for i in layer_range]

    for i, token_range in enumerate(vision_discrete_range):
        x_ticks += np.arange(discrete_range[i][0], discrete_range[i][1] + 1).tolist()
        range_layer_max_prob, layer_words = logitLens_of_vision_tokens(
            model, tokenizer, input_ids, outputs,
            token_range, layer_range,
            logits_warper, logits_processor
        )
        each_range_layer_prob_list.append(range_layer_max_prob)
        each_range_layer_words_list.append(layer_words)

    whole_ranges_layer_prob = np.concatenate(each_range_layer_prob_list, axis=1)

    # plot heatmap
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    im = ax.imshow(
        whole_ranges_layer_prob,
        alpha=0.8,
        )

    # annotate text
    range_flag = 0
    for each_range_layer_words in each_range_layer_words_list:
        for layer_i, each_layer_words in enumerate(each_range_layer_words):
            for col_j, word in enumerate(each_layer_words):
                ax.text(
                    range_flag + col_j, len(layer_range) - 1 - layer_i,
                    word, ha='center', va='center', color='w',
                    fontsize=13, rotation=30,
                )
        range_flag += len(each_layer_words)

    ax.set_xlim(0-0.5, len(x_ticks)-0.5)
    ax.set_xticks([i for i in range(len(x_ticks))])
    ax.set_yticks([i for i in range(len(layer_range))])
    ax.set_xticklabels(x_ticks, fontsize=16)
    ax.set_yticklabels(y_ticks, fontsize=16)
    ax.set_xlabel('Image Tokens Index', fontsize=16)
    ax.set_ylabel('Layers', fontsize=16)
    ax.set_title('Logit Lens of Vision Tokens with Discrete Range', fontsize=16)

    if fig_name is not None:
        plt.savefig(f'./{fig_name}.pdf')
    plt.show()


def plot_VAR_heatmap(avg_data, filename=None):
    # sort heads
    sorted_idx = np.argsort(-avg_data, axis=-1)
    avg_data = np.take_along_axis(avg_data, sorted_idx, axis=-1)

    # plot heatmap
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    im = axes.imshow(
        avg_data, vmin=avg_data.min(),
        vmax=avg_data.max(), cmap='Blues'
    )
    n_layer, n_head = avg_data.shape
    y_label_list = [str(i) for i in range(n_layer)]
    axes.set_yticks(np.arange(0, n_layer, 2))
    axes.set_yticklabels(y_label_list[::-1][::2])
    axes.set_xlabel("Sorted Heads")
    axes.set_ylabel("Layers")
    fig.colorbar(im, ax=axes, shrink=0.4, location='bottom')
    plt.xticks([])
    if filename is not None:
        plt.savefig(filename, dpi=400)
    plt.show()


def show_heatmap_over_image_with_interpolation(
    text, layer_id, head_id, outputs, tokenizer, image,
    vision_token_start, vision_token_end, savefig=True
):
    ''' Show heatmap over the image in i-th head at j-th layer
    Params:
    -------
    text: real or hallucinated object text
    layer_id: int
    head_id: int
    outputs: dict
        additionaly need 'attentions' attribute
    tokenizer: the tokenizer of the model
    image: PIL image
    vision_token_start: int
    vision_token_end: int
    '''
    selected_token_id = tokenizer(text, add_special_tokens=False)["input_ids"][0]
    # the first index is adoptted if there are multiple occurrences
    token_in_generation_idx = torch.nonzero(outputs['sequences'][0][1:] == selected_token_id)[0].item()

    # get row attention matrix
    row_attn = outputs['attentions'][token_in_generation_idx][layer_id].squeeze(0)[head_id].cpu().detach()
    visual_row_attn = row_attn[-1, vision_token_start:vision_token_end].to(torch.float32)
    visual_row_attn = visual_row_attn / visual_row_attn.max()

    # resize
    visual_row_attn = visual_row_attn.reshape(24, 24) # for llava-1.5
    # bilinear interpolation
    attn_over_image = cv2.resize(visual_row_attn.numpy(), (image.size[0], image.size[1]))

    def show_mask_on_image(img, mask):
        img = np.float32(img) / 255
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    np_img = np.array(image)
    # plt.imshow(np_img)
    img_with_attn = show_mask_on_image(np_img, attn_over_image)
    plt.imshow(img_with_attn)
    # turn off axis
    plt.axis('off')
    if savefig:
        plt.savefig(
            f"{text}_heatmap_layer{layer_id}_head{head_id}.png",
            bbox_inches='tight',
            pad_inches=0
        )
    plt.show()