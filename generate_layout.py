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
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np
import clip

from torch.utils.data import Dataset
from torch.nn import Module, Parameter
import time
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
from src.data_utils import collate_fn, PROJECT_ROOT, targetpad_transform
from src.datasets import FashionIQDataset, CIRRDatasetV2, CIRCODataset
from layout_utils.llm_utils import *
from layout_utils.utils import *



def main():
    parser = argparse.ArgumentParser(description="Generate a question-answer instruction tuning dataset.")

    # Mandatory parameter
    parser.add_argument("--sources",default='coco',type=str)

    ####### For physical multi card
    parser.add_argument('--start_idx',default=0,type=int)
    parser.add_argument('--idx',default=0,type=int)
    parser.add_argument('--gpu_num',default=1, type=int)
    
    ###### File Path in Generation
    parser.add_argument('--output_path',default='',type=str)
    parser.add_argument('--llm_path',default='',type=str)
    parser.add_argument('--prompt_config',default='./prompt/prompt_layout_v2.yaml', type=str)
    ###### Layout Generation
    parser.add_argument('--model_type',default='qwen',type=str)
    parser.add_argument('--random_layout',action='store_true',help = 'for ablation study')
    parser.add_argument("--batch_size",default=1,type=int)
    ###### Debug
    parser.add_argument('--visualize',default=False,type=bool)
    parser.add_argument('--visualize_path',default='./visual_layout/',type=str)
    
    ### Dataset Parameters
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    parser.add_argument('--mode',default='circo_test',type=str)
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset", required=False, default='/mnt/data2/comp_data/CIRCO')

    args = parser.parse_args()

    # Placeholder for the dataset generation logic.
    print("Generating Proxy Layout with the following parameters:")
    print("Dataset_mode: ", args.mode)
    print(f'The Generated layout with be saved in {args.output_path}')
    print(f'Generate Layout with {args.model_type}')

    clip_model, clip_preprocess = clip.load('ViT-L/14', device='cpu', jit=False)

    if args.preprocess_type == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
    elif args.preprocess_type == 'clip':
        print('CLIP preprocess pipeline is used')
        preprocess = clip_preprocess
    else:
        raise ValueError("Preprocess type not supported")


    if args.mode == 'fashioniq_shirt':
        # for dress_type in ['shirt', 'dress', 'toptee']:
        relative_val_dataset = FashionIQDataset(args.dataset_path, 'val', ['shirt'], 'relative', preprocess)
        run(relative_val_dataset, preprocess, args)
    if args.mode == 'fashioniq_dress':
        relative_val_dataset = FashionIQDataset(args.dataset_path, 'val', ['dress'], 'relative', preprocess)
        run(relative_val_dataset, preprocess, args)
    if args.mode == 'fashioniq_toptee':
        relative_val_dataset = FashionIQDataset(args.dataset_path, 'val', ['toptee'], 'relative', preprocess)
        run(relative_val_dataset, preprocess, args)

    elif args.mode == 'cirr':
        relative_val_dataset = CIRRDatasetV2(args.dataset_path, 'val', 'relative', preprocess)
        run(relative_val_dataset, preprocess, args)
    elif args.mode == 'cirr_test':
        relative_test_dataset = CIRRDatasetV2(args.dataset_path, 'test', 'relative', preprocess)
        run(relative_test_dataset, preprocess, args)
    elif args.mode == 'circo':
        relative_val_dataset = CIRCODataset(args.dataset_path, 'val', 'relative', preprocess)
        run(relative_val_dataset, preprocess, args)
    elif args.mode == 'circo_test':
        relative_test_dataset = CIRCODataset(args.dataset_path, 'test', 'relative', preprocess)
        run(relative_test_dataset, preprocess, args)
    else:
        run(None, None, args)

def run(dataset, preprocess, args):

    dataset_stastic = {}
    device = torch.device(f'cuda')
    tokenizer, model = load_model_llm(args, device = device)

    prompt_configs = load_config(args)
    prompt_system = prompt_configs['layout']['system_prompt']
    examples = prompt_configs['layout']['examples']

    instance_count = 0
    layout_instance = {}

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    relative_val_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=10,
                                pin_memory=False, collate_fn=collate_fn, shuffle=False)

    # Compute the features
    idx = 0
    inst_id = 0
    seed_idx = 0
    idxx = 0
    for batch in tqdm(relative_val_loader):

        if isinstance(batch, dict):
            reference_names = batch['reference_name']
            relative_captions = batch['relative_caption']
            if 'shared_concept' in batch:
                concept = batch['shared_concept']
                target = []
            else:
                concept = batch['multi_opt'][0]
                target = batch['multi_gpt_opt'][0]
            bsz = len(concept)
        else:
            bsz = 1

        for bs in range(bsz):
            if bs != args.idx:
                continue
            while True: # Generate util has a answer
                random.seed(idx + bsz + args.start_idx + seed_idx)

                if isinstance(concept[bs], tuple):
                    classes = [concept[bs][0]]
                else:
                    classes = [concept[bs]]

                refer = ['image' for i in classes]
                scene = []
                rule = relative_captions[bs]
                if len(target) > 0:
                    tar_ = target[bs]
                    rule = f'generating a image of {tar_}'
                layout = []
                flag, layout_info, messages_hash, question_id = llm_layout(args, classes, rule, scene, layout, refer, model, tokenizer, idx, dataset_stastic, prompt_system, examples, random = args.random_layout)

                if flag == True:
                    break
                seed_idx = seed_idx + 1

            if 'pair_id' in batch.keys():
                question_id = batch['pair_id'].tolist()[bs]
            else:
                question_id = reference_names[bs]

            idx = idx + 1
            if layout_info == None:
                continue
            inst_id = inst_id + 1
            
            if args.visualize:
                if not os.path.exists(args.visualize_path):
                    os.makedirs(args.visualize_path)
                image = np.ones((512,512,3),np.uint8) * 200
                visualize_annotations(image, layout_info, caption = question_id, scontent = None, index = inst_id, step = 0, output_dir=args.visualize_path)
                print(layout_info)

            layout_instance[question_id] = {
                "layout": layout_info,
                "name":question_id,
            }
        idxx = idxx + 1

    json_output_path = os.path.join(args.output_path, f'output_proxy_layout_{args.idx}.json')
    with open(json_output_path,'w') as f:
        json.dump(layout_instance, f)
    
if __name__ == "__main__":
    main()
