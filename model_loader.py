import os
from collections import namedtuple

import torch
import yaml
from CFG import CFGLogits
from constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    IMAGE_TOKEN_INDEX,
    IMAGE_TOKEN_LENGTH,
    MINIGPT4_IMAGE_TOKEN_LENGTH,
    SHIKRA_IMAGE_TOKEN_LENGTH,
    SHIKRA_IMG_END_TOKEN,
    SHIKRA_IMG_START_TOKEN,
)
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from minigpt4.common.eval_utils import init_model
from mllm.models import load_pretrained
from DACD import ContrastiveLogitsProcessor
from ACD import acd
from DCD import dcd
# from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer
def load_model_args_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    ModelArgs = namedtuple("ModelArgs", data["ModelArgs"].keys())
    TrainingArgs = namedtuple("TrainingArgs", data["TrainingArgs"].keys())

    model_args = ModelArgs(**data["ModelArgs"])
    training_args = TrainingArgs(**data["TrainingArgs"])

    return model_args, training_args


def load_llava_model(model_path):
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    return tokenizer, model, image_processor, model


def load_minigpt4_model(cfg_path):
    cfg = MiniGPT4Config(cfg_path)
    model, vis_processor = init_model(cfg)
    # TODO:
    # model.eval()
    return model.llama_tokenizer, model, vis_processor, model.llama_model

def load_shikra_model(yaml_path):
    model_args, training_args = load_model_args_from_yaml(yaml_path)
    model, preprocessor = load_pretrained(model_args, training_args)

    return (
        preprocessor["text"],
        model.to("cuda"),
        preprocessor["image"],
        model.to("cuda"),
    )


class MiniGPT4Config:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.options = None


def load_model(model):
    if model == "llava-1.5":
        model_path = os.path.expanduser("/home/async/data-disk/czh/LLaVA/llava-v1.5-7b")
        return load_llava_model(model_path)


    elif model == "minigpt4":

        cfg_path = "./minigpt4/eval_config/minigpt4_eval.yaml"

        return load_minigpt4_model(cfg_path)

    elif model == "shikra":
        yaml_path = "./mllm/config/config.yml" 
        return load_shikra_model(yaml_path)

    else:
        raise ValueError(f"Unknown model: {model}")


def prepare_llava_inputs(template, query, image, tokenizer):
    image_tensor = image["pixel_values"][0]
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)

    chunks = [q.split("<ImageHere>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id
    )

    img_start_idx = len(token_before[0]) + 1
    img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH
    image_token = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * IMAGE_TOKEN_INDEX
    )

    input_ids = torch.cat([bos, token_before, image_token, token_after], dim=1)
    kwargs = {}
    kwargs["images"] = image_tensor.half()
    kwargs["input_ids"] = input_ids

    return qu, img_start_idx, img_end_idx, kwargs

def prepare_llava_inputs_test(template, query, image, tokenizer):
    image_tensor = image["pixel_values"][0]
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)

    chunks = [q.split("<ImageHere>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id
    )

    img_start_idx = len(token_before[0]) + 1  # Index after [BOS] and text tokens
    img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH


    # Calculate text start and end indices
    text_start_idx = 1  # Index of [BOS] token
    text_end_idx_before_img = img_start_idx  # End index of text before image
    text_start_idx_after_img = img_end_idx  # Start index of text after image
    text_end_idx = text_start_idx_after_img + len(token_after[0])
    image_token = (
            torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
            * IMAGE_TOKEN_INDEX
    )

    input_ids = torch.cat([bos, token_before, image_token, token_after], dim=1)
    kwargs = {}
    kwargs["images"] = image_tensor.half()


    return qu, input_ids,img_start_idx, img_end_idx, text_start_idx, text_end_idx_before_img, text_start_idx_after_img, text_end_idx, kwargs

def prepare_minigpt4_inputs(template, query, image, model):
    image_tensor = image.to("cuda")
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)

    img_embeds, atts_img = model.encode_img(image_tensor.to("cuda"))
    inputs_embeds, attention_mask = model.prompt_wrap(
        img_embeds=img_embeds, atts_img=atts_img, prompts=qu
    )

    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device=inputs_embeds.device)
        * model.llama_tokenizer.bos_token_id
    )
    bos_embeds = model.embed_tokens(bos)
    atts_bos = attention_mask[:, :1]
    text_before_img = qu[0].split("<ImageHere>")[0]
    text_after_img = qu[0].split("<ImageHere>")[1]

    text_before_img_ids = model.llama_tokenizer(
        text_before_img, return_tensors="pt", add_special_tokens=False
    ).input_ids.shape[-1]

    text_after_img_ids = model.llama_tokenizer(
        text_after_img, return_tensors="pt", add_special_tokens=False
    ).input_ids.shape[-1]

    # add 1 for bos token
    img_start_idx = (
        model.llama_tokenizer(
            qu[0].split("<ImageHere>")[0], return_tensors="pt", add_special_tokens=False
        ).input_ids.shape[-1]
        + 1
    )

    img_end_idx = img_start_idx + MINIGPT4_IMAGE_TOKEN_LENGTH
    text_start_idx = 1  # Index of [BOS] token
    text_end_idx_before_img = text_start_idx + text_before_img_ids  # End index of text before image
    text_start_idx_after_img = img_end_idx + 1 # Start index of text after image
    text_end_idx = text_start_idx_after_img + text_after_img_ids


    inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

    kwargs = {}
    kwargs["inputs_embeds"] = inputs_embeds
    kwargs["attention_mask"] = attention_mask


    return qu, img_start_idx, img_end_idx, text_start_idx, text_end_idx_before_img, text_start_idx_after_img, text_end_idx, kwargs


def prepare_shikra_inputs(template, query, image, tokenizer):
    image_tensor = image["pixel_values"][0]

    replace_token = DEFAULT_IMAGE_PATCH_TOKEN * SHIKRA_IMAGE_TOKEN_LENGTH
    qu = [template.replace("<question>", q) for q in query]
    qu = [p.replace("<ImageHere>", replace_token) for p in qu]

    input_tokens = tokenizer(
        qu, return_tensors="pt", padding="longest", add_special_tokens=False
    ).to("cuda")

    bs = len(query)
    bos = torch.ones([bs, 1], dtype=torch.int64, device="cuda") * tokenizer.bos_token_id
    input_ids = torch.cat([bos, input_tokens.input_ids], dim=1)

    img_start_idx = torch.where(input_ids == SHIKRA_IMG_START_TOKEN)[1][0].item()
    img_end_idx = torch.where(input_ids == SHIKRA_IMG_END_TOKEN)[1][0].item()

    kwargs = {}
    kwargs["input_ids"] = input_ids
    kwargs["images"] = image_tensor.to("cuda")

    return qu, img_start_idx, img_end_idx, kwargs



class ModelLoader:
    def __init__(self, model_name,load_in_8bit=False, load_in_4bit=False):
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.tokenizer = None
        self.vlm_model = None
        self.llm_model = None
        self.image_processor = None
        self.load_model()

    def load_model(self):
        if self.model_name == "llava-1.5":
            model_path = os.path.expanduser("/home/async/data-disk/czh/LLaVA/llava-v1.5-7b")
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_llava_model(model_path)
            )

        elif self.model_name == "minigpt4":
            cfg_path = "./minigpt4/eval_config/minigpt4_eval.yaml"
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_minigpt4_model(cfg_path)
            )

        elif self.model_name == "shikra":
            yaml_path = "./mllm/config/config.yml"
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_shikra_model(yaml_path)
            )
        # elif self.model_name == "qwen":
        #     model_path = os.path.expanduser("/home/ubuntu/data-disk/czh/Qwen2.5-VL-7B-Instruct")
        #     self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
        #         load_qwen_model(model_path)
        #     )

        else:
            raise ValueError(f"Unknown model: {self.model}")

    def prepare_inputs_for_model(self, template, query, image):
        if self.model_name == "llava-1.5":
            questions,input_ids, img_start_idx, img_end_idx, text_start_idx, text_end_idx_before_img, text_start_idx_after_img, text_end_idx, kwargs = prepare_llava_inputs_test(
                template, query, image, self.tokenizer
            )
            self.img_start_idx = img_start_idx
            self.img_end_idx = img_end_idx
            self.text_start_idx = text_start_idx
            self.text_end_idx_before_img = text_end_idx_before_img
            self.text_start_idx_after_img = text_start_idx_after_img
            self.text_end_idx = text_end_idx

            return questions, input_ids, kwargs
        elif self.model_name == "minigpt4":
            questions, img_start_idx, img_end_idx,text_start_idx, text_end_idx_before_img, text_start_idx_after_img, text_end_idx, kwargs = prepare_minigpt4_inputs(
                template, query, image, self.vlm_model
            )
        elif self.model_name == "shikra":
            questions, img_start_idx, img_end_idx, kwargs = prepare_shikra_inputs(
                template, query, image, self.tokenizer
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.img_start_idx = img_start_idx
        self.img_end_idx = img_end_idx
        self.text_start_idx = text_start_idx
        self.text_end_idx_before_img = text_end_idx_before_img
        self.text_start_idx_after_img = text_start_idx_after_img
        self.text_end_idx = text_end_idx


        return questions,kwargs

    def init_dcd_processor(self,kwargs, questions, gamma=1.1, beam=1, start_layer=0, end_layer=32, use_attn=True,
                           alpha=0.2,b=0.2, use_cfg=True,model_loader=None):
        tokens = self.tokenizer(
            questions,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=True,
        ).input_ids.to("cuda")

        prompt_tokens = tokens.repeat(beam, 1)

        logits_processor = dcd(
            kwargs=kwargs,
            guidance_scale=gamma,
            prompt_tokens=prompt_tokens.to("cuda"),
            model=self.llm_model,
            start_layer=start_layer,
            end_layer=end_layer,
            use_attn=use_attn,
            alpha=alpha,
            b = b,
            use_cfg=use_cfg,
            model_loader=model_loader,
        )

        return logits_processor



    def decode(self, output_ids):
        # get outputs
        if self.model_name == "llava-1.5":
            # replace image token by pad token
            output_ids = output_ids.clone()
            output_ids[output_ids == IMAGE_TOKEN_INDEX] = torch.tensor(
                0, dtype=output_ids.dtype, device=output_ids.device
            )

            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]

        elif self.model_name == "minigpt4":
            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [
                text.split("###")[0].split("Assistant:")[-1].strip()
                for text in output_text
            ]

        elif self.model_name == "shikra":
            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]

        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return output_text
