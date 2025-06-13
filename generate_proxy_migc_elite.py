import argparse
import asyncio
import json
import logging
import re
from tqdm import tqdm
from enum_definitions import *
from data_definitions import *
from typing import Iterator
import yaml
import transformers
import torch
from diffusers import EulerDiscreteScheduler
from MIGC.migc.migc_utils import seed_everything,offlinePipelineSetupWithSafeTensor
from MIGC.migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np
from predictor.simple_inference import MLP, normalized
import clip
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from layout_utils.utils import *
from torch.utils.data import Dataset
from torch.nn import Module, Parameter
import time
import PIL
from PIL import Image
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import numpy as np
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING
from typing import Optional, Tuple, Union
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, LMSDiscreteScheduler
from migc.migc_arch import MIGC, NaiveFuser
import copy
from pycocotools import mask as mask_utils
import os

from packaging import version
if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

class MapperLocal(nn.Module):
    def __init__(self,
         input_dim: int,
         output_dim: int,
    ):
        super(MapperLocal, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                              nn.LayerNorm(1024),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1024, 1024),
                                                              nn.LayerNorm(1024),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:])
            hidden_states += (hidden_state.unsqueeze(0),)
        hidden_states = torch.cat(hidden_states, dim=0).mean(dim=0)
        return hidden_states

class Mapper(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
    ):
        super(Mapper, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state, )
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states

def get_sup_mask(mask_list):
    or_mask = np.zeros_like(mask_list[0])
    for mask in mask_list:
        or_mask += mask
    or_mask[or_mask >= 1] = 1
    sup_mask = 1 - or_mask
    return sup_mask


class MIGCProcessorELITE(nn.Module):
    def __init__(self, config, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.not_use_migc = config['not_use_migc']
        self.naive_fuser = NaiveFuser()
        if not self.not_use_migc:
            self.migc = MIGC(config['C'])

    def __call__(
            self,
            attn,
            hidden_states = None,
            encoder_hidden_states=None,
            attention_mask=None,
            prompt_nums=[],
            bboxes=[],
            ith=None,
            embeds_pooler=None,
            timestep=None,
            height=512,
            width=512,
            MIGCsteps=20,
            NaiveFuserSteps=-1,
            ca_scale=None,
            ea_scale=None,
            sac_scale=None,
            local_inj_info = {},
        ):
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        
        instance_num = len(bboxes[0])
        if ith > MIGCsteps:
            not_use_migc = True
        else:
            not_use_migc = self.not_use_migc
        is_vanilla_cross = (not_use_migc and ith > NaiveFuserSteps)
        if instance_num == 0:
            is_vanilla_cross = True
 
        is_cross = encoder_hidden_states is not None

        if is_cross:
            context_tensor = encoder_hidden_states
        else:
            context_tensor = hidden_states


        # Only Need Negative Prompt and Global Prompt.
        if is_cross and is_vanilla_cross:
            context_tensor = context_tensor[:2, ...]

        # In this case, we need to use MIGC or naive_fuser, so we copy the hidden_states_cond (instance_num+1) times for QKV
        if is_cross and not is_vanilla_cross:
            hidden_states_uncond = hidden_states[[0], ...]
            hidden_states_cond = hidden_states[[1], ...].repeat(instance_num + 1, 1, 1)
            hidden_states = torch.cat([hidden_states_uncond, hidden_states_cond])

        hidden_states_local = hidden_states.clone()

        # QKV Operation of Vanilla Self-Attention or Cross-Attention
        query = attn.to_q(hidden_states)

        if is_cross:
            if 'LOCAL_INDEX' in local_inj_info:
                index_local_ = local_inj_info["LOCAL_INDEX"]
                key = self.to_k_global(context_tensor)
                value = self.to_v_global(context_tensor)
                key_o = attn.to_k(context_tensor)
                value_o = attn.to_v(context_tensor)
                key_final = [key_o[:1,...]]
                value_final = [value_o[:1,...]]
                for idx in range(len(index_local_)):
                    index = index_local_[idx]
                    if index!=-1:
                        value_final.append(value[idx + 1:idx + 2,...])
                        key_final.append(key[idx + 1:idx + 2,...])
                    else:
                        value_final.append(value_o[idx + 1:idx + 2,...])
                        key_final.append(key_o[idx + 1:idx + 2,...])
                    idx = idx + 1
                key = torch.cat(key_final,dim = 0)
                value = torch.cat(value_final,dim=0)
            else:
                key = attn.to_k(context_tensor)
                value = attn.to_v(context_tensor)
        else:
            key = attn.to_k(context_tensor)
            value = attn.to_v(context_tensor)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)  # 48 4096 77
        self.attnstore(attention_probs, is_cross, self.place_in_unet)
        hidden_states = torch.bmm(attention_probs, value)

        if len(bboxes[0]) > 0:
            guidance_masks = []
            for bbox in bboxes[0]:  
                guidance_mask = np.zeros((height, width))
                w_min = int(width * bbox[0])
                w_max = int(width * bbox[2])
                h_min = int(height * bbox[1])
                h_max = int(height * bbox[3])
                guidance_mask[h_min: h_max, w_min: w_max] = 1.0
                guidance_masks.append(guidance_mask[None, ...])

            w = int((hidden_states.shape[1]) ** 0.5)
            guidance_masks = np.concatenate(guidance_masks, axis=0)
            guidance_masks = guidance_masks[None, ...]
            guidance_masks = torch.from_numpy(guidance_masks).float().to(hidden_states.device)
            guidance_masks = F.interpolate(guidance_masks, (w, w), mode='bilinear')  # (1, instance_num, H, W)


        tmp = []
        if is_cross and "LOCAL" in local_inj_info and len(bboxes[0]) > 0:
            index_local_ = local_inj_info["LOCAL_INDEX"]
            idx = 0
            for index_local in index_local_:
                # print(index_local_)
                if index_local!=-1:
                    index_local = index_local[None,...]
                    # Perform cross attention with the local context
                    query_local = attn.to_q(hidden_states_local[idx + 1:idx + 2,...])
                    key_local = self.to_k_local(local_inj_info["LOCAL"][idx].to(hidden_states_local.device))
                    value_local = self.to_v_local(local_inj_info["LOCAL"][idx].to(hidden_states_local.device))

                    query_local = attn.head_to_batch_dim(query_local)
                    key_local = attn.head_to_batch_dim(key_local)
                    value_local = attn.head_to_batch_dim(value_local)

                    attention_scores_local = torch.matmul(query_local, key_local.transpose(-1, -2))
                    attention_scores_local = attention_scores_local * attn.scale
                    # attention_scores_local = attention_scores_local
                    attention_probs_local = attention_scores_local.softmax(dim=-1)

                    # To extract the attmap of learned [w]
                    index_local = index_local.reshape(index_local.shape[0], 1).repeat((1, attn.heads)).reshape(-1)
                    attention_probs_clone = attention_probs.clone().permute((0, 2, 1))
                    attention_probs_mask = attention_probs_clone[torch.arange(index_local.shape[0]), index_local]
                    # Normalize the attention map
                    attention_probs_mask = attention_probs_mask.unsqueeze(2) / attention_probs_mask.max()

                    if "LAMBDA" in local_inj_info:
                        _lambda = local_inj_info["LAMBDA"]
                    else:
                        _lambda = 1

                    attention_probs_local = attention_probs_local * attention_probs_mask * _lambda
                    hidden_states_tmp = torch.matmul(attention_probs_local, value_local)
                    tmp.append(hidden_states_tmp)
                else:
                    tmp.append(hidden_states[(idx + 1) * 8: (idx + 2) * 8,...])
                    
                    # value_local_list.append(value_local)
                idx = idx + 1
            tmp_m = torch.cat(tmp, dim=0)
            hidden_states[8:,...] = hidden_states[8:,...] * 0.5 + tmp_m * 0.5

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        ###### Self-Attention Results ######
        if not is_cross:  
            return hidden_states

        ###### Vanilla Cross-Attention Results ######
        if is_vanilla_cross:
            return hidden_states
        
        ###### Cross-Attention with MIGC ######
        assert (not is_vanilla_cross)
        hidden_states_uncond = hidden_states[[0], ...]  # torch.Size([1, HW, C])
        cond_ca_output = hidden_states[1: , ...].unsqueeze(0)  # torch.Size([1, 1+instance_num, 5, 64, 1280])
        guidance_masks = []
        in_box = []
        # Construct Instance Guidance Mask
        for bbox in bboxes[0]:  
            guidance_mask = np.zeros((height, width))
            w_min = int(width * bbox[0])
            w_max = int(width * bbox[2])
            h_min = int(height * bbox[1])
            h_max = int(height * bbox[3])
            guidance_mask[h_min: h_max, w_min: w_max] = 1.0
            guidance_masks.append(guidance_mask[None, ...])
            in_box.append([bbox[0], bbox[2], bbox[1], bbox[3]])
        
        # Construct Background Guidance Mask
        sup_mask = get_sup_mask(guidance_masks)
        supplement_mask = torch.from_numpy(sup_mask[None, ...])
        supplement_mask = F.interpolate(supplement_mask, (height//8, width//8), mode='bilinear').float()
        supplement_mask = supplement_mask.to(hidden_states.device)  # (1, 1, H, W)
        
        guidance_masks = np.concatenate(guidance_masks, axis=0)
        guidance_masks = guidance_masks[None, ...]
        guidance_masks = torch.from_numpy(guidance_masks).float().to(cond_ca_output.device)
        guidance_masks = F.interpolate(guidance_masks, (height//8, width//8), mode='bilinear')  # (1, instance_num, H, W)

        in_box = torch.from_numpy(np.array(in_box))[None, ...].float().to(cond_ca_output.device)  # (1, instance_num, 4)

        other_info = {}
        other_info['image_token'] = hidden_states_cond[None, ...]
        other_info['context'] = context_tensor[1:, ...]  # instance num, 77, 768
        other_info['box'] = in_box
        other_info['context_pooler'] = embeds_pooler  # (instance_num, 1, 768)
        other_info['supplement_mask'] = supplement_mask
        other_info['attn2'] = None
        other_info['attn'] = attn
        other_info['height'] = height
        other_info['width'] = width
        other_info['ca_scale'] = ca_scale
        other_info['ea_scale'] = ea_scale
        other_info['sac_scale'] = sac_scale
        if not not_use_migc:
            hidden_states_cond, fuser_info = self.migc(cond_ca_output,
                                            guidance_masks,
                                            other_info=other_info,
                                            return_fuser_info=True)
        else:
            hidden_states_cond, fuser_info = self.naive_fuser(cond_ca_output,
                                            guidance_masks,
                                            other_info=other_info,
                                            return_fuser_info=True)
        hidden_states_cond = hidden_states_cond.squeeze(1)

        hidden_states = torch.cat([hidden_states_uncond, hidden_states_cond])
        return hidden_states


@add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
def inj_forward_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify either input_ids")

    r_input_ids = input_ids['input_ids']
    if 'inj_embedding' in input_ids:
        inj_embedding = input_ids['inj_embedding']
        inj_index = input_ids['inj_index']
    else:
        inj_embedding = None
        inj_index = None

    input_shape = r_input_ids.size()
    r_input_ids = r_input_ids.view(-1, input_shape[-1])


    inputs_embeds = self.embeddings.token_embedding(r_input_ids)
    new_inputs_embeds = inputs_embeds.clone()
    if inj_embedding is not None:
        emb_length = inj_embedding.shape[1]
        for bsz, idx in enumerate(inj_index):
            lll = new_inputs_embeds[bsz, idx+emb_length:].shape[0]
            new_inputs_embeds[bsz, idx+emb_length:] = inputs_embeds[bsz, idx+1:idx+1+lll]
            new_inputs_embeds[bsz, idx:idx+emb_length] = inj_embedding[bsz]

    hidden_states = self.embeddings(input_ids=r_input_ids, position_ids=position_ids, inputs_embeds=new_inputs_embeds)

    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # expand attention_mask

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=r_input_ids.device), r_input_ids.to(torch.int).argmax(dim=-1)
    ]

    if not return_dict:
        # print('xixi')
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )

def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask

@torch.no_grad()
def validation(example, pipe, image_encoder, mapper, mapper_local, device, guidance_scale, prompt, bboxes, migc_param, unconde_input, layout_info,sam_predictor, grounding_dino_model, glip_demo, llambda=1, num_steps=50):
    # 获取example中的内容
    scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    uncond_embeddings = unconde_input['embed']
    uncond_embed_pooler = unconde_input['pooler']

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    prompt_nums = [0] * len(prompt)

    for i, _ in enumerate(prompt):
        prompt_nums[i] = len(_)

    placeholder_idx = example["index"]
    image_f = example['pixel_values_clip']
    input_ids = example['input_ids']
    embeds_pooler = []
    encoder_hidden_states = []
    inj_embedding_local_list = []
    for idx in range(len(input_ids)):
        feature = image_f[idx]

        if feature!=None:
            # import pdb;pdb.set_trace()
            image = F.interpolate(feature[None,...], (224, 224), mode='bilinear')
            image_features = image_encoder(image, output_hidden_states=True)
            image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16]]
            image_embeddings = [emb.detach() for emb in image_embeddings]
            inj_embedding = mapper(image_embeddings).half()

            inj_embedding = inj_embedding[:, 0:1, :]

            pooler_embeds = pipe.text_encoder({'input_ids': example["input_ids"][idx].cuda()},return_dict = True)['pooler_output']
            encoder_hidden_states_ = pipe.text_encoder({'input_ids': example["input_ids"][idx].cuda(),
                                                "inj_embedding": inj_embedding,
                                                "inj_index": placeholder_idx[idx][None, ...]})[0]

            image_obj = F.interpolate(example["pixel_values_obj"][idx][None,...], (224, 224), mode='bilinear')
            image_features_obj = image_encoder(image_obj, output_hidden_states=True)
            image_embeddings_obj = [image_features_obj[0], image_features_obj[2][4], image_features_obj[2][8],
                                    image_features_obj[2][12], image_features_obj[2][16]]
            image_embeddings_obj = [emb.detach() for emb in image_embeddings_obj]

            inj_embedding_local = mapper_local(image_embeddings_obj)
            mask = F.interpolate(example["pixel_values_seg"][idx][None,...], (16, 16), mode='nearest')
            mask = mask[:, 0].reshape(mask.shape[0], -1, 1)
            inj_embedding_local = inj_embedding_local * mask
            embeds_pooler.append(pooler_embeds[:,None, ...])
            encoder_hidden_states.append(encoder_hidden_states_)
            inj_embedding_local_list.append(inj_embedding_local)
        else:
            output = pipe.text_encoder({'input_ids': example["input_ids"][idx].cuda()},return_dict = True)
            encoder_hidden_states_ = output[0]
            pooler_embeds = output['pooler_output']
            embeds_pooler.append(pooler_embeds[:,None,...])
            encoder_hidden_states.append(encoder_hidden_states_)
            inj_embedding_local_list.append(None)

    encoder_hidden_states = torch.cat(encoder_hidden_states, dim = 0)
    embeds_pooler = torch.cat(embeds_pooler,dim = 0)

    encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states],dim=0)
    local_inj_info = {
        "LOCAL": inj_embedding_local_list,
        "LOCAL_INDEX": placeholder_idx,
        "LAMBDA": llambda,
    }

    failure = 0
    skip_mode = True
    nums = 0
    while True:
        seed = random.randint(0, 10000000000000000000)
        seed_everything(seed)
        image_info = pipe(prompt, bboxes, num_inference_steps=50, guidance_scale=7.5, 
                            MIGCsteps=migc_param['MIGCsteps'], NaiveFuserSteps=migc_param['NaiveFuserSteps'], aug_phase_with_and=False, return_inital_latents = False, prompt_embeds = encoder_hidden_states, embeds_pooler = embeds_pooler, local_inj_info = local_inj_info).images[0]
        flag = filter_image_position(image_info, layout_info, sam_predictor, grounding_dino_model, glip_demo, index = failure)
        if flag or skip_mode:
            return image_info
        else:
            failure = failure + 1
            if failure > 3:
                skip_mode = True

def load_input(prompt_list, ref_list, image_path_list, tokenizer, mask_list):

    def process(image):
        img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def get_tensor_clip(normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)

    interpolation = {
        "linear": PIL_INTERPOLATION["linear"],
        "bilinear": PIL_INTERPOLATION["bilinear"],
        "bicubic": PIL_INTERPOLATION["bicubic"],
        "lanczos": PIL_INTERPOLATION["lanczos"],
    }['bicubic']

    example = {}

    # Step 1. Reading the image that need to be injected
    idx = 0
    index = []
    input_ids = []
    text = []
    pixel_values_obj = []
    pixel_values_clip = []
    pixel_values_seg = []
    pixel_values = []
    placeholder_string = '*'
    for ref in ref_list:
        prompt = prompt_list[idx]
        input_ids.append(tokenizer(
                        [prompt],
                        padding="max_length",
                        truncation=True,
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids)
        text.append(prompt)
        if ref == 'text':
            index.append(-1)
            pixel_values_obj.append(None)
            pixel_values_clip.append(None)
            pixel_values_seg.append(None)
            pixel_values.append(None)
        else:
            mask = mask_list[idx]
            if mask is None:
                index.append(-1)
                pixel_values_obj.append(None)
                pixel_values_clip.append(None)
                pixel_values_seg.append(None)
                pixel_values.append(None)
                continue
            else:
                image_path = image_path_list[idx]
                placeholder_index = 0
                words = prompt.strip().split(' ')
                for _, word in enumerate(words):
                    if word == placeholder_string:
                        placeholder_index = _ + 1

                index.append(torch.tensor(placeholder_index))
                image_path = image_path_list[idx]
                image = Image.open(image_path)
                if not image.mode == "RGB":
                    image = image.convert("RGB")

                image_np = np.array(image)
                object_tensor = image_np * mask
                pixel_values.append(process(image_np))

                pixel_values_obj.append(get_tensor_clip()(Image.fromarray(object_tensor.astype('uint8')).resize((224, 224), resample=interpolation)))
                pixel_values_clip.append(get_tensor_clip()(Image.fromarray(image_np.astype('uint8')).resize((224, 224), resample=interpolation)))
                ref_seg_tensor = Image.fromarray(mask.astype('uint8') * 255)
                ref_seg_tensor = get_tensor_clip(normalize=False)(ref_seg_tensor)
                pixel_values_seg.append(torch.nn.functional.interpolate(ref_seg_tensor.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0))
        idx = idx + 1
    example['text'] = text
    example['index'] = index
    example['pixel_values_obj'] = pixel_values_obj
    example['pixel_values_clip'] = pixel_values_clip
    example['pixel_values_seg'] = pixel_values_seg
    example['pixel_values'] = pixel_values
    example['input_ids'] = input_ids

    return example

def th2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)

def equip_elite_mapper( device, mapper_model_path, mapper_local_model_path, pipe):

    # Load models and create wrapper for stable diffusion
    for _module in pipe.text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text

    mapper = Mapper(input_dim=1024, output_dim=768)

    mapper_local = MapperLocal(input_dim=1024, output_dim=768)

    image_encoder = CLIPVisionModel.from_pretrained("/mnt/data2/liyou/ckpt/clip-vit-large-patch14")

    for _name, _module in pipe.unet.named_modules():
        
        if _module.__class__.__name__ == "Attention":
            if 'attn1' in _name: continue

            shape = _module.to_k.weight.shape
            to_k_global = nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_k', to_k_global)

            shape = _module.to_v.weight.shape
            to_v_global = nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_v', to_v_global)

            to_v_local = nn.Linear(shape[1], shape[0], bias=False)
            mapper_local.add_module(f'{_name.replace(".", "_")}_to_v', to_v_local)

            to_k_local = nn.Linear(shape[1], shape[0], bias=False)
            mapper_local.add_module(f'{_name.replace(".", "_")}_to_k', to_k_local)

    mapper.load_state_dict(torch.load(mapper_model_path, map_location='cpu'))
    # mapper.half()

    mapper_local.load_state_dict(torch.load(mapper_local_model_path, map_location='cpu'))
    # mapper_local.half()

    for _name, _module in pipe.unet.named_modules():
        if 'attn1' in _name: continue
        # print(_name)
        # print(_module.__class__.__name__)
        if _module.__class__.__name__ == "MIGCProcessorELITE":
            act_name = '.'.join(_name.split('.')[:-1])
            _module.add_module('to_k_global', mapper.__getattr__(f'{act_name.replace(".", "_")}_to_k'))
            _module.add_module('to_v_global', mapper.__getattr__(f'{act_name.replace(".", "_")}_to_v'))
            _module.add_module('to_v_local', getattr(mapper_local, f'{act_name.replace(".", "_")}_to_v'))
            _module.add_module('to_k_local', getattr(mapper_local, f'{act_name.replace(".", "_")}_to_k'))
            # _module = _module.cuda()

    mapper.eval()
    mapper_local.eval()
    return pipe, image_encoder, mapper, mapper_local

def load_model_generation(args, device = 'cuda'):
    aes_predictor = None
    clip_model2 = None
    clip_preprocess = None
    sam_predictor = None
    grounding_dino_model = None
    glip_demo = None
            
    if args.sample_image:
        if args.sd1x_path.split('/')[-1].split('.')[-1] == 'safetensors':
            pipe = offlinePipelineSetupWithSafeTensor(sd_safetensors_path=args.sd1x_path)
        else:
            pipe = StableDiffusionMIGCPipeline.from_pretrained(args.sd1x_path)
        pipe.attention_store = AttentionStore()
        from MIGC.migc.migc_utils import load_migc
        load_migc(pipe.unet , pipe.attention_store,
                args.migc_ckpt_path, attn_processor=MIGCProcessorELITE)
        # load_migc(pipe.unet , pipe.attention_store,
        #         args.migc_ckpt_path, attn_processor=MIGCProcessor)
        # if args.with_lora:
        #     pipe.load_lora_weights(args.lora_path)
        pipe = pipe.to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        aes_predictor = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        s = torch.load("./predictor/sac+logos+ava1-l14-linearMSE.pth") 
        aes_predictor.load_state_dict(s)

        aes_predictor.to(device)
        aes_predictor.eval()

        clip_model2, clip_preprocess = clip.load("./weights/ViT-L-14.pt", device=device)  #RN50x64 

        GROUNDING_DINO_CONFIG_PATH = "./Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT_PATH = args.dino_ckpt

        grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=device)


        SAM_ENCODER_VERSION = "vit_h"
        SAM_CHECKPOINT_PATH = args.sam_ckpt

        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device = 'cuda')
        sam_predictor = SamPredictor(sam)

        config_file = "./GLIP/configs/pretrain/glip_Swin_L.yaml"
        weight_file = "./weights/glip_large_model.pth"
        cfg.local_rank = 0
        cfg.num_gpus = 1
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

        glip_demo = GLIPDemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
            show_mask_heatmaps=False
        )

    return pipe, aes_predictor, clip_model2, clip_preprocess, sam_predictor, grounding_dino_model, glip_demo

def main():
    parser = argparse.ArgumentParser(description="Generate a question-answer instruction tuning dataset.")
    ####### arguments for migc
    parser.add_argument('--sample_image',default=True,type=bool)
    parser.add_argument('--migc_ckpt_path',default = './weights/MIGC_SD14.ckpt',type=str)
    parser.add_argument('--sd1x_path',default = './weights/realisticVisionV60B1_v60B1VAE.safetensors',type = str)
    parser.add_argument("--aug_phase_with_and", type=bool,default=False)
    parser.add_argument("--NaiveFuserSteps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--MIGCsteps", type=int, default=25)
    ####### Reannotation
    parser.add_argument('--dino_ckpt',default='./weights/groundingdino_swint_ogc.pth',type=str)
    parser.add_argument('--sam_ckpt',default='./weights/sam_vit_h_4b8939.pth',type=str)

    ####### For test or train
    parser.add_argument('--idx',default=0,type=int)
    parser.add_argument('--gpu_num',default=1, type=int)
    ####### For layout
    parser.add_argument('--layout_file',default='',type=str)
    parser.add_argument('--mask_file',default='./prompt/relative_mask_circo_test.json',type=str)
    parser.add_argument('--image_source',default='',type=str)
    parser.add_argument('--mode',default='circo',type=str)
    ###### Output
    parser.add_argument('--output_path',default='',type=str)
    parser.add_argument('--aug_caption',default='high quality image',type=str)
    parser.add_argument('--align_with_gt',default=False,type=bool,help='Adding addtional detection results')
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--img_per_mode',default=1,type=int)
    parser.add_argument('--pure_bg',action='store_true')
    
    args = parser.parse_args()

    run(args)


def run(args):

    # Load model for generation
    device = torch.device(f'cuda')
    pipe, aes_predictor, clip_model2, clip_preprocess, sam_predictor, grounding_dino_model, glip_demo = load_model_generation(args, device = device)
    mapper_model_path = './weights/global_mapper.pt'
    mapper_local_model_path = './weights/local_mapper.pt'

    pipe, image_encoder, mapper, mapper_local = equip_elite_mapper(device, mapper_model_path, mapper_local_model_path, pipe)
    pipe = pipe.to('cuda')

    
    guidance_scale = args.guidance_scale
    migc_param = {
        'MIGCsteps': args.MIGCsteps,
        'NaiveFuserSteps': args.NaiveFuserSteps,
        'negative_prompt': 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    }

    uncond_input = pipe.tokenizer(
        ['worst quality, low quality, bad anatomy'] * 1,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )

    uncond_embeddings_dict = pipe.text_encoder({'input_ids':uncond_input.input_ids.to(device)}) # [1, 77, 768]
    uncond_embeddings = uncond_embeddings_dict[0]
    uncond_embed_pooler = uncond_embeddings_dict['pooler_output']
    unconde_input = {
        'embed':uncond_embeddings,
        'pooler':uncond_embed_pooler,
    }

    with open(args.layout_file, 'r') as f:
        layout_instance = json.load(f)

    with open(args.mask_file, 'r') as f:
        mask_info = json.load(f)
    

    # Create directory path for output
    output_dir = args.output_path
    addition_path = ['combined']

    for add_ in addition_path:
        new_path = os.path.join(output_dir, add_)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
    
    # Skip the cases that have been generated
    exi_id = defaultdict(int)
    for img_name in os.listdir(os.path.join(output_dir, 'combined')):
        idx = img_name.split('_')[1]
        exi_id[idx] += 1

    total_infer_list = list(layout_instance.keys())
    need_inst = [
        str(id_) for id_ in total_infer_list 
        if id_ not in exi_id or exi_id[id_] < args.img_per_mode
    ]

    need_num = len(need_inst)
    need_num_pn = need_num // args.gpu_num
    

    if args.idx == args.gpu_num - 1:
        instance_key = need_inst[args.idx * need_num_pn : ]
    else:
        instance_key = need_inst[args.idx * need_num_pn : (args.idx + 1) * need_num_pn]

    print(f'Need to generate {len(instance_key)} instance')

    for img_id in tqdm(instance_key):
        if args.mode == 'cirr':
            source_name = layout_instance[img_id]['name'].split('.')[0]
        else:
            source_name = img_id

        ref_image_path = args.image_source
        if args.mode == 'circo':
            img_name = source_name.zfill(12) + '.jpg'
        else:
            img_name = source_name + '.jpg'

        source_image_path = os.path.join(ref_image_path, img_name)
        ref_image = Image.open(source_image_path).convert('RGB')
        ref_image = np.array(ref_image)
        layout_info = layout_instance[img_id]['layout']
        mask = None
        if img_id in mask_info:
            mask_ = mask_info[img_id]
            mask = mask_utils.decode(mask_)
            mask = mask[:,:,0:1]
            mask = np.repeat(mask, 3, axis=2)


        prompt_final_combine = [[]]
        bboxes_combine = [[]]
        ref_list_combine = []
        image_path_list_combine = []
        mask_list_combine = []
        has_global = False
        for inst in layout_info:
            if 'ref' in inst:
                ref = inst['ref']
            else:
                ref = 'text'
            if 'desc' in inst:
                desc = inst['desc']
            else:
                desc = ''
            if 'label' in inst:
                label = inst['cate']
            else:
                label = ''
            if 'bbox' in inst:
                bbox = inst['bbox']
            else:
                continue
            if 'is_scene' in inst and inst['is_scene']:
                if has_global == True:
                    continue
                has_global = True
                if args.pure_bg:
                    desc = 'pure white background.'
                prompt_final_combine[0].insert(0, desc)
                ref_list_combine.insert(0, 'text')
                image_path_list_combine.insert(0, '_')
                mask_list_combine.insert(0, None)
                continue
            bboxes_combine[0].append(bbox)
            if ref == 'image':
                prompt_final_combine[0].append(desc)
                prompt_final_combine[0].append(f'a * {label}')
                ref_list_combine.append('text')
                ref_list_combine.append('image')
                if 'CIRCO' in args.image_source:
                    image_path_list_combine.append('_')
                    image_path_list_combine.append(os.path.join(args.image_source, source_name.zfill(12) + '.jpg'))
                else:
                    image_path_list_combine.append('_')
                    image_path_list_combine.append(os.path.join(args.image_source, source_name + '.jpg'))

                bboxes_combine[0].append(bbox)

                if img_id in mask_info:
                    mask_list_combine.append(None)
                    mask_list_combine.append(mask)
                else:
                    detections = detect_on_image(ref_image, [inst['cate']], sam_predictor, grounding_dino_model)
                    bbox = detections.xyxy.tolist()
                    if len(bbox) > 0:
                        mask_result = segment_on_bbox(ref_image, bbox[0], sam_predictor)
                        mask = mask_utils.decode(mask_result)
                        mask = mask[:,:,0:1]
                        mask = np.repeat(mask, 3, axis=2)
                    else:
                        print('no object detect')
                        mask = None
                    mask_list_combine.append(None)
                    mask_list_combine.append(mask)
            else:
                prompt_final_combine[0].append(desc)
                image_path_list_combine.append('_')
                ref_list_combine.append('text')
                mask_list_combine.append(None)

        if has_global == False:
            prompt_final_combine[0].insert(0, 'a high quality image')
            ref_list_combine.insert(0, 'text')
            image_path_list_combine.insert(0, '_')
            mask_list_combine.insert(0, None)
        
        example_combine = load_input(prompt_final_combine[0], ref_list_combine, image_path_list_combine, pipe.tokenizer, mask_list_combine)
        example_input = {
            'combined':example_combine,
        }
        prompt_input = {
            'combined':prompt_final_combine,
        }
        bbox_input = {
            'combined':bboxes_combine,
        }

        nums = 0
        if args.debug:
            import shutil
            inst_debug_path = os.path.join(debug_path, img_id)
            if not os.path.exists(inst_debug_path):
                os.makedirs(inst_debug_path)
            origin_image_path = args.image_source
            useful_info = cr[str(img_id)]
            source_image_path = os.path.join(origin_image_path, str(img_id).zfill(12) + '.jpg')
            t_sip = os.path.join(inst_debug_path, 'source.jpg')
            shutil.copy(source_image_path, t_sip)
            source_target_path = os.path.join(origin_image_path, str(useful_info['target_img_id']).zfill(12) + '.jpg')
            t_sip = os.path.join(inst_debug_path, 'target.jpg')
            shutil.copy(source_target_path, t_sip)
            source_txt_path = os.path.join(inst_debug_path, 'instruction.json')
            output_info = {
                'instruction': useful_info['relative_caption'],
                'layout': layout_info
            }
            with open(source_txt_path,'w') as f:
                json.dump(output_info, f)
            output_path = inst_debug_path
        else:
            output_path = output_dir
        
        with tqdm(total=3 * args.img_per_mode) as pbar:
            for idx in tqdm(range(args.img_per_mode)):
                prompt = prompt_final_combine
                bbox = bboxes_combine
                example_ = example_combine
                syn_images = validation(example_, pipe, image_encoder, mapper, mapper_local, device, guidance_scale, prompt, bbox, migc_param, unconde_input, layout_info,sam_predictor, grounding_dino_model, glip_demo, llambda=float(1.0))
                
                if not args.debug:
                    out_path = os.path.join(output_dir, 'combined')
                    curr_id = exi_id[img_id]
                    save_name = os.path.join(out_path, f'combined_{img_id}_{idx + curr_id}.jpg')
                else:
                    out_path = output_path
                    save_name = os.path.join(out_path, f'combined_{img_id}_{idx + curr_id}.jpg')
                syn_images.save(save_name)
                pbar.update(1)
                    
        if args.debug:
            inst_n = inst_n + 1
            if inst_n > 10:
                break
    

if __name__ == "__main__":
    main()
