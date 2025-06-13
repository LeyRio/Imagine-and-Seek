import json
import pickle
from args import args_define
from typing import List, Tuple, Dict

import clip
import open_clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_ import CIRCODataset
from utils import extract_image_features, device, collate_fn, PROJECT_ROOT, targetpad_transform
import os
import PIL



@torch.no_grad()
def extract_aug_features(dataset_dir, clip_model):

    data_list = os.listdir(dataset_dir)
    index_features = []
    index_names = []
    for img in data_list:
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')
        images = images.to(device)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features.append(batch_features.cpu())
            index_names.extend(names)

    index_features = torch.vstack(index_features)
    return index_features, index_names



@torch.no_grad()
def circo_generate_test_submission_file(dataset_path: str, clip_model_name: str, 
                                        preprocess: callable,
                                        submission_name: str) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    # Load the CLIP model
    if 'L' in clip_model_name:
        type_ = 'L'
    else:
        type_ = 'G'
        
    if clip_model_name == 'ViT-g-14':
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', device=device, pretrained='./weights/CLIP-ViT-g-14-laion2B-s12B-b42K/open_clip_pytorch_model.bin')
    else:
        clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Compute the index features
    classic_test_dataset = CIRCODataset(dataset_path, 'test', 'classic', preprocess)
    feature_path = os.path.join(args.feature_dir, f'circo_test_{type_}.npz')
    if os.path.exists(feature_path):
        feature_content = torch.load(feature_path)
        index_features = feature_content['feature']
        index_names = feature_content['name']
    else:
        index_features, index_names = extract_image_features(classic_test_dataset, clip_model)
        save_content = {
            'feature':index_features,
            'name':index_names,
        }
        torch.save(save_content, feature_path)

    relative_test_dataset = CIRCODataset(dataset_path, 'test', 'relative', preprocess)

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, clip_model, index_features,
                                                           index_names, args.nums_caption)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'circo'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)

@torch.no_grad()
def prepare_aug_feature(relative_test_dataset, clip_model, aug_dir = '', type='L'):
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=16,
                                      pin_memory=False, collate_fn=collate_fn, shuffle=False)
    list_info = {
        'combined':{},
    }
    if type == 'G':
        tokenizer = open_clip.get_tokenizer('ViT-g-14')
    else:
        tokenizer = clip.tokenize

    aug_info = ['combined']
    for aug_ in aug_info:
        aug_path = os.path.join(aug_dir, aug_)
        for img_name_ in os.listdir(aug_path):
            if img_name_.split('_')[len(aug_.split('_'))] in list_info[aug_]:
                list_info[aug_][img_name_.split('_')[len(aug_.split('_'))]].append(img_name_)
            else:
                list_info[aug_][img_name_.split('_')[len(aug_.split('_'))]]=[img_name_]

    path = f'{aug_dir}/aug_features_{type}.npz'
    with open(args.layout_path, 'r') as f:
        aug_text = json.load(f)
    if os.path.exists(path):
        aug_features = torch.load(path)
    else:
        aug_features = {
            'combined':{},
            'txt':{},
        }
        for batch in tqdm(relative_test_loader):
            reference_names = batch['reference_name']
            curr_feature = {
                'combined':[],
            }
            for idx in reference_names:
                for aug_ in aug_info:
                    img_name_list = list_info[aug_][idx]
                    f_list = []
                    for img_name in img_name_list:
                        aug_dir_ = os.path.join(aug_dir, aug_)
                        img_path = os.path.join(aug_dir_, img_name)
                        img_f = relative_test_dataset.preprocess(PIL.Image.open(img_path))
                        f_list.append(img_f)
                    img_f = torch.stack(f_list)
                    curr_feature[aug_].append(img_f)
            
            for aug_ in aug_info:
                with torch.no_grad():
                    curr_feature[aug_] = torch.utils.data.dataloader.default_collate(curr_feature[aug_])
                    feature = []
                    for i in range(curr_feature[aug_].shape[0]):
                        batch_features_iter = clip_model.encode_image(curr_feature[aug_][i,...].cuda())
                        feature.append(batch_features_iter[None,...])
                    batch_features = torch.cat(feature, dim=0)
                    for i in range(batch_features.shape[0]):
                        aug_features[aug_][reference_names[i]] = batch_features[i : i + 1,...]

            curr_text_feature = []
            for idx in reference_names:
                text_features_list = []
                aug_inst = aug_text[idx]['layout']
                for inst in aug_inst:
                    if type == 'G':
                        tokenized_input_captions = tokenizer(inst['desc'], context_length=77).to(device)
                    else:
                        tokenized_input_captions = tokenizer(inst['desc'], context_length=77,truncate=True).to(device)
                    text_features = clip_model.encode_text(tokenized_input_captions)
                    text_features_list.append(text_features)
                text_features = torch.cat(text_features_list, dim=0)
                aug_features['txt'][idx] = text_features
        save_file_path = f'{aug_dir}/aug_features_{type}.npz'
        torch.save(aug_features, save_file_path)

    source_path = f'{aug_dir}/source_features_{type}.npz'
    if os.path.exists(source_path):
        source_features = torch.load(source_path)
    else:
        source_features = {}
        for batch in tqdm(relative_test_loader):
            reference_names = batch['reference_name']
            curr_source = []
            for idx in reference_names:
                img_dir_ = os.path.join(args.dataset_path, 'COCO2017_unlabeled/unlabeled2017') 
                image_name = idx.zfill(12) + '.jpg'
                image_path = os.path.join(img_dir_, image_name)
                img_f = relative_test_dataset.preprocess(PIL.Image.open(image_path))
                curr_source.append(img_f)
            
            with torch.no_grad():
                curr_source = torch.utils.data.dataloader.default_collate(curr_source)
                batch_features = clip_model.encode_image(curr_source.cuda())
                for i in range(batch_features.shape[0]):
                    source_features[reference_names[i]] = batch_features[i : i + 1,...]

        save_file_path = f'{aug_dir}/source_features_{type}.npz'
        torch.save(source_features, save_file_path)
    aug_features['source'] = source_features

    return aug_features


def get_feature(x, mode='mean'):
    N,C = x.shape
    if mode == 'random':
        return  x[torch.randint(0, N, (1,))]
    elif mode == 'mean':
        return x.mean(dim=0, keepdim=True)
    elif mode == 'max':
        max_output, _ = x.max(dim=0, keepdim=True)
        return max_output
    elif mode == 'first':
        return x[0].unsqueeze(0)
    else:
        return x.mean(dim=0, keepdim=True)
    
def circo_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRCODataset,
                                    use_momentum_strategy=False, debiased_id=-1) -> [torch.Tensor, List[List[str]]]:
    """
    Generate the test prediction features for the CIRCO dataset given the pseudo tokens
    """

    # Create the test dataloader
    # num_workers
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=16,
                                      pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    refer_names = {}
    caps = {}
    query_ids_list = []
    if args.type == 'G':
        tokenizer = open_clip.get_tokenizer('ViT-g-14')
    else:
        tokenizer = clip.tokenize

    with_aug = args.with_aug and not use_momentum_strategy
    if with_aug:
        aug_features_stack = {
            'composed':[],
            'pure_text':[],
            'combined':[],
            'source':[],
        }
        aug_text_feature = []
        final_aug_f = {}

        # aug_dir = '/mnt/data0/liyou/output/output_comimage/testv2/images'
        aug_dir = args.aug_dir
        aug_info = ['combined', 'source']
        aug_features = prepare_aug_feature(relative_test_dataset, clip_model, type=args.type, aug_dir = aug_dir)
        

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        relative_captions = batch['relative_caption']
        query_ids = batch['query_id']
        blip2_caption = batch['blip2_caption_{}'.format(args.caption_type)]
        gpt_caption = batch['gpt_caption_{}'.format(args.caption_type)]
        multi_caption = batch['multi_{}'.format(args.caption_type)]
        multi_gpt_caption = batch['multi_gpt_{}'.format(args.caption_type)]
        # import pdb;pdb.set_trace()
        if with_aug:
            curr_feature = {
                'composed':[],
                'pure_text':[],
                'combined':[],
                'source':[],
            }
            curr_text_feature = []

            
            for aug_ in aug_info:
                for idx in reference_names:
                    curr_feature[aug_].append(aug_features[aug_][idx])
                    
                aug_features_stack[aug_].append(torch.cat(curr_feature[aug_], dim=0).cpu())
            for idx in reference_names:
                    
                curr_text_feature.append(get_feature(aug_features['txt'][idx], args.aug_type))
                
            aug_text_feature.append(torch.cat(curr_text_feature, dim=0))

            
 
        if args.is_gpt_caption:
            if args.multi_caption:
                if use_momentum_strategy:
                    if debiased_id != -1:
                        input_captions = multi_caption[debiased_id]
                    else:
                        input_captions = multi_caption
                else:
                    if debiased_id != -1:
                        input_captions = multi_gpt_caption[debiased_id]
                    else:
                        input_captions = multi_gpt_caption
            else:
                if use_momentum_strategy:
                    input_captions = blip2_caption
                else:
                    input_captions = [f"{caption}" for caption in gpt_caption]
        else:
            if args.multi_caption and args.is_rel_caption:
                input_captions = multi_caption
                for i in range(len(input_captions)): 
                    input_captions[i] = [f"a photo of {input_captions[i][inx]} that {relative_captions[inx]}" for inx in range(len(input_captions[i]))]
            else:
                input_captions = [f"a photo that {caption}" for caption in relative_captions]

        if args.multi_caption and debiased_id == -1:
            text_features_list = []
            source_text_features_list = []
            for cap in input_captions:
                tokenized_input_captions = tokenizer(cap, context_length=77,truncate=True).to(device)
                text_features = clip_model.encode_text(tokenized_input_captions)
                text_features_list.append(text_features)
            text_features_list = torch.stack(text_features_list)
            text_features = torch.mean(text_features_list, dim=0)

        else:
            if args.type != 'G':
                tokenized_input_captions = tokenizer(input_captions, context_length=77,truncate=True).to(device)
            else:
                tokenized_input_captions = tokenizer(input_captions, context_length=77).to(device)
            # import pdb;pdb.set_trace()
            text_features = clip_model.encode_text(tokenized_input_captions)
        predicted_features = F.normalize(text_features)
        predicted_features_list.append(predicted_features)
        query_ids_list.extend(query_ids)
        for idx in range(len(query_ids)):
            index = query_ids[idx]
            refer_names[index] = reference_names[idx]
            caps[index] = relative_captions[idx]

    predicted_features = torch.vstack(predicted_features_list)
    if with_aug:
        for aug_ in aug_info:
            final_aug_f[aug_] = torch.vstack(aug_features_stack[aug_])
        aug_text_feature = torch.vstack(aug_text_feature)
        aug_text_feature = F.normalize(aug_text_feature)
        # import pdb;pdb.set_trace()
        predicted_features = {
            'predicted':predicted_features,
            'aug':final_aug_f,
            'aug_txt':aug_text_feature,
        }

    else:
        predicted_features = {'predicted':predicted_features}

    return predicted_features, query_ids_list, refer_names, caps

def circo_generate_test_dict(relative_test_dataset: CIRCODataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str], nums_caption) \
        -> Dict[str, List[str]]:
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    if args.is_gpt_predicted_features:
        if args.use_debiased_sample:
            predicted_features_list = []
            for i in range(nums_caption):
                predicted_features = torch.load('feature/debiased/gpt_predicted_features_{}.pt'.format(i))
                predicted_features_list.append(predicted_features)
            query_ids = np.load('feature/query_ids.npy')

        else:
            predicted_features = torch.load('feature/gpt_predicted_features.pt')
            query_ids = np.load('feature/query_ids.npy')
    else:
        if args.use_debiased_sample:
            predicted_features_list = []
            for i in range(nums_caption):
                predicted_features_dict, query_ids, refer_names, caps = circo_generate_test_predictions(clip_model, relative_test_dataset, debiased_id=i)
                if isinstance(predicted_features_dict, dict):
                    predicted_features = predicted_features_dict['predicted']
                    if args.with_aug and 'aug' in predicted_features_dict:
                        aug_features = predicted_features_dict['aug']
                        if 'aug_txt' in predicted_features_dict:
                            aug_feature_t = predicted_features_dict['aug_txt']
                torch.save(predicted_features, 'feature/debiased/gpt_predicted_features_{}.pt'.format(i))
                predicted_features_list.append(predicted_features)
        else:
            predicted_features_dict, query_ids, refer_names, caps = circo_generate_test_predictions(clip_model, relative_test_dataset)
            if isinstance(predicted_features_dict, dict):
                predicted_features = predicted_features_dict['predicted']
                if args.with_aug and 'aug' in predicted_features_dict:
                    aug_features = predicted_features_dict['aug']
                    if 'aug_txt' in predicted_features_dict:
                        aug_feature_t = predicted_features_dict['aug_txt']

            if args.features_save_path:
                np.save('feature/query_ids.npy', query_ids)
                torch.save(predicted_features, 'feature/gpt_predicted_features.pt')
    with open('./circo_info.json', 'w') as f:
        instance = {'caps':caps, 'names':refer_names}
        json.dump(instance, f)
    
    # Normalize the features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)
    

    if args.use_momentum_strategy:
        if args.is_blip_predicted_features:
            if args.use_debiased_sample:
                blip_predicted_features_list = []
                for i in range(nums_caption):
                    blip_predicted_features = torch.load('feature/debiased/blip_predicted_features_{}.pt'.format(i))
                    blip_predicted_features_list.append(blip_predicted_features)
            else:
                blip_predicted_features = torch.load('feature/blip_predicted_features.pt')
        else:
            if args.use_debiased_sample:
                blip_predicted_features_list = []
                for i in range(nums_caption):
                    blip_predicted_features_dict, _, _, _ = circo_generate_test_predictions(clip_model, relative_test_dataset, True, debiased_id=i)
                    if isinstance(blip_predicted_features_dict, dict):
                        blip_predicted_features = blip_predicted_features_dict['predicted']
                    torch.save(blip_predicted_features, 'feature/debiased/blip_predicted_features_{}.pt'.format(i))
                    blip_predicted_features_list.append(blip_predicted_features)
            else:
                blip_predicted_features_dict, _, _, _ = circo_generate_test_predictions(clip_model, relative_test_dataset, True)
                if isinstance(blip_predicted_features_dict, dict):
                    blip_predicted_features = blip_predicted_features_dict['predicted']
                if args.features_save_path:
                    torch.save(blip_predicted_features, 'feature/blip_predicted_features.pt')
        
        num_N = 5
        if args.use_debiased_sample:
            neg_diff_val = []
            if args.with_aug:
                diff_aug_text = []
                sim_aug = {}
                final_sim = []
                final_feature_aug = []

                for aug in ['combined']:
                    aug_features[aug] = aug_features[aug].float().to(device)
                    aug_features[aug] = F.normalize(aug_features[aug], dim=-1)
                    tmp_sim = []
                    curr_feature = []
                    for i in range(num_N):
                        curr_aug_features = aug_features[aug][:,i,:]
                        curr_sim = curr_aug_features @ index_features.T
                        tmp_sim.append(curr_sim)
                        curr_feature.append(curr_aug_features)
                    sim_aug[aug] = torch.mean(torch.stack(tmp_sim), dim=0)
                    aug_feature = torch.mean(torch.stack(curr_feature), dim=0)
                    final_sim.append(sim_aug[aug])
                    final_feature_aug.append(aug_feature)
                source_feature = aug_features['source']
                source_feature = F.normalize(source_feature, dim=-1)
                aug_similarity = torch.mean(torch.stack(final_sim), dim = 0)
                aug_img_feature = torch.mean(torch.stack(final_feature_aug), dim = 0)
                aug_similarity_t = aug_feature_t @ index_features.T

                sim_combine = aug_similarity * aug_similarity_t
                
            sim_after = []
            total_diff = []
            robust_direction_list = []
            for i in range(nums_caption):
                gpt_features = predicted_features_list[i]
                blip_features = blip_predicted_features_list[i]

                robust_direction_list.append(gpt_features - blip_features)

                similarity_after = gpt_features @ index_features.T
                sim_after.append(similarity_after)
                similarity_before = blip_features @ index_features.T
                diff = similarity_after - similarity_before
                total_diff.append(diff.detach().cpu())
                diff[diff > 0] = 0
                diff = -diff
                diff = torch.topk(diff, dim=-1, k=50).values
                sum_diff = torch.sum(diff)
                neg_diff_val.append(sum_diff.item())

            
            
            similarity_after_ = torch.stack(sim_after)
            similarity_after_ = torch.mean(similarity_after_, dim=0)
            neg_diff_val_tensor = torch.tensor(neg_diff_val).float().to(device)
            debiased_weight = torch.softmax(neg_diff_val_tensor / torch.max(neg_diff_val_tensor) / args.debiased_temperature, 0)
            predicted_features_tensor = torch.stack(predicted_features_list)
            if 0:
                debiased_features = torch.mean(predicted_features_tensor, dim=0)
            else:
                debiased_features = torch.sum(predicted_features_tensor * debiased_weight.unsqueeze(1).unsqueeze(2), dim=0)

            robust_direction = torch.mean(torch.stack(robust_direction_list), dim=0)
            aug_feature = torch.mean(aug_features['combined'], dim=1)
            
            # robust_aug_img = 1.0 * source_feature.float().to(device) + 1.0 * (source_feature.max() / robust_direction.max()) * robust_direction.float().to(device) + 1.0 * (source_feature.max() / aug_img_feature.max()) * aug_img_feature
            robust_aug_img = args.s_w * (aug_img_feature.max() / source_feature.max()) * source_feature.float().to(device) + args.t_w * (aug_img_feature.max() / robust_direction.max()) * robust_direction.float().to(device) + args.a_w * aug_img_feature
            robust_aug_similarity = robust_aug_img.float().to(device) @ index_features.T.float().to(device)
            
            similarity = debiased_features  @ index_features.T
            if args.with_aug:
                total_diff = torch.stack(total_diff)
                total_diff = total_diff.mean(dim=0).cuda()
                scale = args.fusion_weight # 0.3 for G 0.65 for L
                similarity = scale * similarity + (1 - scale) * robust_aug_similarity * similarity
        else:
            similarity_after = predicted_features @ index_features.T
            similarity_before = blip_predicted_features @ index_features.T

            diff = similarity_after - similarity_before

            similarity = similarity_after + args.momentum_factor * diff

    # Compute the similarity
    else:
        similarity = predicted_features @ index_features.T
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}

    return queryid_to_retrieved_images


args = args_define.args

def main():
    if args.eval_type in ['LDRE-B', 'LDRE-L', 'LDRE-G']:
        if args.eval_type == 'LDRE-B':
            clip_model_name = 'ViT-B/32'
        elif args.eval_type == 'LDRE-L':
            clip_model_name = 'ViT-L/14'
        else:
            clip_model_name = 'ViT-g-14'

        if clip_model_name == 'ViT-g-14':
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='./weights/CLIP-ViT-g-14-laion2B-s12B-b42K/open_clip_pytorch_model.bin')
        else:
            clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

        if args.preprocess_type == 'targetpad':
            print('Target pad preprocess pipeline is used')
            preprocess = targetpad_transform(1.25, 224)
        elif args.preprocess_type == 'clip':
            print('CLIP preprocess pipeline is used')
            preprocess = clip_preprocess
        else:
            raise ValueError("Preprocess type not supported")

        relative_test_dataset = CIRCODataset(args.dataset_path, 'test', 'relative', preprocess)
        clip_model = clip_model.float().to(device)
        
    args.exp_name = 'crico_base'
    print(f"Eval type = {args.eval_type} \t exp name = {args.exp_name} \t")

    circo_generate_test_submission_file(args.dataset_path, clip_model_name, preprocess, args.submission_name)




if __name__ == '__main__':
    main()
