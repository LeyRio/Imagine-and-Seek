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

# from dataset import coco, open_images
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np
import openai
import clip
from layout_utils.utils import *
from layout_utils.llm_utils import *
from torch.utils.data import Dataset
from torch.nn import Module, Parameter
import time
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle
import sys

class OpenAI_API(object):
    def __init__(self, api_key):
        self.api_key = api_key

    def get_response(self, prompt, max_tokens=5, temperature=0.9, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=["\n", " Human:", " AI:"]):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            max_tokens=max_tokens,
        )
        return response

def load_model_llm(args, max_token = 4096, device = 'cuda'):
    tokenizer = None
    model = None
            
    if args.model_type == 'openai':
        import openai
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        openai.proxy = {
            'http': 'http://127.0.0.1:7890',
            'https': 'http://127.0.0.1:7890'
            }
        openai.Model.list()
        model = OpenAI_API(openai.api_key)
    elif args.model_type == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.llm_path, device_map="auto", trust_remote_code=True).eval()
    elif args.model_type == 'deepseek':
        from openai import OpenAI
        model = OpenAI(api_key="", base_url="https://api.deepseek.com")

    return tokenizer, model

def get_chat_prompt(prompt_system, inputs, examples):
    messages = [
        {"role": "system", "content": prompt_system}
    ]
    for sample in examples:
        messages.append({"role": "user", "content": sample["input"]})
        messages.append({"role": "assistant", "content": sample["output"]})
    messages.append({"role": "user", "content": inputs})
    # print(messages)

    return messages

def apply_llm(args, messages, model, tokenizer, max_token = 4096):
    responses = []
    for prompt in messages:
        if args.model_type == 'openai':
            result = model.get_response(prompt, max_tokens=max_token)
            response = result['choices'][0]['message']['content']
            responses.append(response)
        elif args.model_type == 'deepseek':
            response = model.chat.completions.create(
                model="deepseek-chat",
                messages=prompt,
                stream=False
            )
            responses.append(response.choices[0].message.content)
        elif args.model_type == 'qwen':
            device = "cuda" 
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            attention_mask = torch.ones(model_inputs.input_ids.shape,dtype=torch.long,device=device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_token,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
    return responses

def run_llm_inferene(args, model, tokenizer, inputs, prompt_system, examples):
    if isinstance(inputs, str):
        inputs = [inputs]
    prompts = [get_chat_prompt(prompt_system, input_, examples) for input_ in inputs]
    outputs = apply_llm(args, prompts, model, tokenizer)
    return outputs


def llm_layout(args, classes, rule, scene, layout, refer, model, tokenizer, image_idx, dataset_stastic, prompt_system, examples, random = False):
    if random:
        layout_info = generate_layout_random(args, classes, rule, scene, dataset_stastic)
    else:
        basic_layout_info = get_advise(args, classes, rule, scene, layout, refer, model, tokenizer, prompt_system, examples)
        if len(basic_layout_info) <2:
            return False, None, None, None
        layout_info = enrich_layout(basic_layout_info, dataset_stastic = dataset_stastic)
        
    messages_hash = hashlib.sha256(json.dumps(layout_info, sort_keys=True).encode('utf-8')).hexdigest()
    image_id = f"IPCIR_{messages_hash}_{image_idx}"
    return True, layout_info, messages_hash, image_id

def generate_layout_random(args, classes, rule, scene, dataset_stastic):
    layout_info = get_random(args, classes, dataset_stastic, '')
    return layout_info

def get_advise(args, classes, rule, scene, layout, refer, model, tokenizer, prompt_system, examples):

    instruction_input = get_simple_inputs(classes, rule = rule, scene = scene, Initial_Layout = layout, refer = refer)
    basic_input = [get_chat_prompt(prompt_system, instruction_input, examples)]
    reasoning_output = apply_llm(args, basic_input, model, tokenizer)

    reasoning_info = reasoning_output[0].split('\n')
    advise_info = [i.strip().lower() for i in reasoning_info if i!='' and '##' in i]
    basic_layout_info = []
    has_scene = False
    final_scene = ''

    for advise_ in advise_info:
        instance = {}
        advise_list = [i for i in advise_.split('[##') if i!='']

        if len(advise_list) == 1 and 'scene' in advise_list[0]:
            key_ = advise_list[0].split(':')[0].strip().lower()
            value_ = advise_list[0].split(':')[-1].strip().lower().split('##]')[0].strip().lower()
            value_ = value_.split('.')[0]
            final_scene = value_
        else:
            for inst in advise_list:
                key_ = inst.split(':')[0].strip().lower()
                value_ = inst.split(':')[-1].strip().lower().split('##]')[0].strip().lower()
                if 'bbox' in key_:
                    try:
                        bbox = []
                        b_ = value_.split(',')
                        for _ in b_:
                            _ = float(_.strip().rstrip(";"))
                            bbox.append(_)
                        value_ = bbox
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area == 1.0:
                            has_scene = True
                            instance['is_scene'] = True
                        else:
                            instance['is_scene'] = False
                    except:
                        continue
                elif key_ == 'from':
                    if '0' in value_:
                        instance['from'] = 0
                    else:
                        instance['from'] = 1
                
                instance[key_] = value_
            if 'label' in instance and 'cate' in instance and 'desc' in instance and 'bbox' in instance and 'ref' in instance:
                basic_layout_info.append(instance)
            
    if has_scene == False and final_scene!='':
        basic_layout_info.append({'label':final_scene, 'cate':final_scene, 'is_scene':True, 'desc':final_scene,'bbox':[0.0,0.0,1.0,1.0],'ref':'text', 'size':5})
    return basic_layout_info


def get_simple_inputs(reference_name, rule = '', scene = [], Initial_Layout = [], refer = []):
    if len(refer) == 0:
        refer = ['text' for i in reference_name]
    input_list = []
    # if len(scene) > 0:
    object_list = ', '.join(reference_name)
    input_string = ''
    if len(scene) > 0:
        input_string = input_string + f'Scene: {scene[0]}'

    if len(reference_name) > 0:
        input_string = input_string + '\nObject: '
    idx = 0
    for obj in reference_name:
        input_string = input_string + f'Label : {obj}, reference: {refer[idx]}'
        if len(Initial_Layout) > idx and Initial_Layout[idx] is not None:
            input_string = input_string + f', Layout: {Initial_Layout[idx]}, '
        idx = idx + 1
    if rule!='':
        input_string = input_string + f'\nLayout Rule: {rule}'

    return input_string



