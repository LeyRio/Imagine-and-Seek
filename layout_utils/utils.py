import hashlib
import json
from textwrap import dedent
import random
from typing import Callable, Dict
import supervision as sv


from data_definitions import *
import yaml
import random
import torch
import groundingdino.datasets.transforms as T
import numpy as np
import torchvision
from segment_anything import sam_model_registry, SamPredictor
import cv2
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re
from torch.nn.functional import relu, sigmoid
import math
from scipy.stats import truncnorm
import collections
import os


color_dict = {
    'red':[{'Lower':np.array([0,175,20]),'Upper':np.array([5,255,255])},{'Lower':np.array([159,50,70]),'Upper':np.array([180,255,255])}],
    'blue':{'Lower':np.array([90, 50, 70]),'Upper':np.array([128, 255, 255])},
    'yellow':{'Lower':np.array([20,100,100]),'Upper':np.array([30,255,255])},
    'green':{'Lower':np.array([36, 50, 70]),'Upper':np.array([89, 255, 255])},
    'black':{'Lower':np.array([0,0,0]),'Upper':np.array([180,255,40])},
    'white':{'Lower':np.array([0,0,99]),'Upper':np.array([180,62,255])},
    'brown':{'Lower':np.array([6,43,35]),'Upper':np.array([25,255,255])},
    'purple':{'Lower':np.array([128, 50, 70]),'Upper':np.array([158, 255, 255])},
    'orange':{'Lower':np.array([6, 100, 20]),'Upper':np.array([19, 255, 255])},
    'gray':{'Lower':np.array([0, 0, 40]),'Upper':np.array([180, 18, 255])},
    'pink':{'Lower':np.array([143, 50, 20]),'Upper':np.array([165, 255, 255])},
}

################################# utils for visualization #############################################################

def draw_bbox(image, bbox, label, color):
    height, width, _ = image.shape
    x1, y1, x2, y2 = bbox
    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    
    if y1 - text_height - baseline < 0:
        text_y = y1 + text_height + baseline
    else:
        text_y = y1 - baseline
    
    image = cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

def parse_label(label):
    color_dict = {
        'red': (0,0,255),
        'blue':(255, 0, 0),'green':(0, 255, 0),'orange':(0, 165, 255),'black':(0,0,0),'gray':(190, 190, 90),
        'yellow':(0, 255, 255),'white':(255, 255, 255),'purple':(240, 32, 160),'pink':(203, 192, 255),'brown':(57, 76, 139),
    }
    label_list = label.split(' ')
    color_list = [
        'red', 'blue', 'green', 'orange','black', 'gray','yellow','white','purple','pink','brown'
    ]
    target_color_bbox = random.choice(color_list)
    target_color_mask = random.choice(color_list)
    for i in color_list:
        if i in label_list:
            target_color_bbox = i
            target_color_mask = i
    return color_dict[target_color_bbox], color_dict[target_color_mask]

def draw_instance(item, image_bbox):
    label = item['label']
    bbox = item['bbox']

    if bbox[3] > 1.0 or bbox[2] > 1.0:
        bbox = [bbox[0] / 512, bbox[1] / 512, bbox[2] / 512, bbox[3] / 512]
    color_bbox, color_mask = parse_label(label)
    image_bbox = draw_bbox(image_bbox, bbox, label, color_bbox)
    return image_bbox
        
def visualize_annotations(image, layout, caption, scontent, index, step, output_dir = './visualize_output/'):
    output_dir = os.path.join(output_dir, f'{step}/{index}_{caption}/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    import copy
    image_bbox = copy.deepcopy(image)
    for _ in layout:
        image_bbox = draw_instance(_, image_bbox)
        
    output_path = os.path.join(output_dir, f'annotation_bbox.png')
    plt.imsave(output_path, image_bbox)

################################# utils for proxy generation #############################################################

def calculate_iou(box1, box2):

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # 没有重叠

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        # print(masks)
        index = np.argmax(scores)
        maskk = np.asfortranarray(masks[index])
        maskk = mask_utils.encode(maskk)
        maskk['counts'] = maskk['counts'].decode('utf-8')
        result_masks.append(maskk)
    return result_masks

def detect_on_image(image, CLASSES, sam_predictor, grounding_dino_model, box_t = 0.25, text_t = 0.25):
    segment_label = {}

    # We set both the box_t and text_t as 0.25 NMS 0.8
    BOX_THRESHOLD = box_t
    TEXT_THRESHOLD = text_t
    NMS_THRESHOLD = 0.8

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    return detections

def draw_mask(image, mask_rle, color = None):
    if color == None:
        color_list = [
            'red', 'blue', 'green', 'orange','black', 'gray','yellow','white','purple','pink','brown'
        ]
        target_color = random.choice(color_list)
        color_dict = {
            'red': (0,0,255),
            'blue':(255, 0, 0),'green':(0, 255, 0),'orange':(0, 165, 255),'black':(0,0,0),'gray':(190, 190, 90),
            'yellow':(0, 255, 255),'white':(255, 255, 255),'purple':(240, 32, 160),'pink':(203, 192, 255),'brown':(57, 76, 139),
        }
        color = color_dict[target_color]
    mask = mask_utils.decode(mask_rle)[:,:,0]
    image_copy = np.array(image)
    for c in range(3):
        image_copy[:, :, c] = np.where(mask == 1, image_copy[:, :, c] * 0.5 + color[c] * 0.5, image_copy[:, :, c])
    return Image.fromarray(image_copy)

def filter_image_position(image, layout, sam_predictor, grounding_dino_model, glip, index = 0):
    class_dict = {}
    for inst in layout:
        if ('is_scene' in inst and inst['is_scene']) or ('from' in inst and inst['from'] == "0"):
            continue
        if inst['cate'] not in class_dict:
            class_dict[inst['cate']] = []
        class_dict[inst['cate']].append(inst['bbox'])

    image_arr = np.array(image)
    for classes in class_dict:
        detect_info = []
        detections = detect_on_image(image_arr, [classes], sam_predictor, grounding_dino_model)
        bbox = detections.xyxy.tolist()
        confi = detections.confidence.tolist()
        for _ in range(len(confi)):
            if confi[_] > 0.3:
                detect_info.append((bbox[_], True))
        result, _ = glip.run_on_web_image(image_arr, classes, 0.5)
        glip_bbox = _.bbox.numpy().tolist()
        used_bbox = [False for i in glip_bbox]
        layout_gt = class_dict[classes]
            
        for gt_bbox in layout_gt:
            gt_bbox = [i * 512 for i in gt_bbox]
            max_iou_dino = 0.0
            max_index_dino = -1
            max_iou_glip = 0.0
            max_index_glip = -1

            for _ in range(len(detect_info)):
                bbox, flag = detect_info[_]
                if not flag:
                    continue
                iou = calculate_iou(bbox, gt_bbox)
                if iou > max_iou_dino:
                    max_index_dino = _
                    max_iou_dino = iou

            for _ in range(len(glip_bbox)):
                bbox = glip_bbox[_]
                flag = used_bbox[_]
                if not flag:
                    iou = calculate_iou(bbox, gt_bbox)
                    if iou > max_iou_glip:
                        max_index_glip = _
                        max_iou_glip = iou
            
            if max_iou_dino < 0.25 and max_iou_glip < 0.25:
                return False
            if max_iou_dino > 0.25:
                detect_info[max_index_dino] = (None, False)
            if max_iou_glip > 0.25:
                used_bbox[max_index_glip] = True
    return True

def save_image(final_image, final_image_name, upscaler = None):
    print(f'save at {final_image_name}')
    final_image.save(final_image_name)


####################################### utils for Layout generation ##############################################################

def load_config(args):
    prompt_configs_fg = None
    with open(args.prompt_config, 'r') as file:
        prompt_configs = yaml.safe_load(file)
        
    return prompt_configs

def enrich_layout(basic_layout_info, dataset_stastic = None, shift_step=10, preturb_ratio = 0.25, preturb_ratio_mv = 0.1):
    for inst_id in range(len(basic_layout_info)):
        inst = basic_layout_info[inst_id]
        placed_boxes = []
        if 'is_scene' in inst and inst['is_scene']:
            continue
        for _ in range(len(basic_layout_info)):
            inst_ = basic_layout_info[_]
            if ('is_scene' in inst_ and inst_['is_scene']) or inst_id == _:
                continue
            if 'bbox' in inst:
                placed_boxes.append(inst_['bbox'])

        label = inst['cate']
        bbox = inst['bbox']
        if len(bbox)!= 4:
            continue
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        area = w * h
        if h == 0 or w == 0:
            continue
        ratio = w / h
        # Step1. Perturb on weight and height
        if label in dataset_stastic:
            w_mean = dataset_stastic[label]['length_mean']
            w_std = dataset_stastic[label]['length_std']
            r_mean = dataset_stastic[label]['aspect_ratio_mean']
            r_std = dataset_stastic[label]['aspect_ratio_std']
            w_samples = np.random.normal(loc=w_mean, scale=w_std, size=100)
            filtered_samples = w_samples[(w_samples >= 0.1) & (w_samples <= 0.7)]
            w_mean_filtered = float(np.mean(filtered_samples))
            r_samples = np.random.normal(loc=r_mean, scale=r_std, size=100)
            filtered_samples = r_samples[(r_samples >= 0.1) & (r_samples <= 0.7)]
            r_mean_filtered = float(np.mean(filtered_samples))
            adjust_weight_a = np.random.uniform(0.7, 0.8)
            adjust_weight_w = np.random.uniform(0.1, 0.3)
            try:
                size = size_dict[str(round(float(inst['size'])))]
                if size > 5:
                    size = 5
                adjust_weight_a = ((size / 5)) * adjust_weight_a
                adjust_weight_w = ((size / 5)) * adjust_weight_w
            except:
                adjust_weight_a = adjust_weight_a
                adjust_weight_w = adjust_weight_w
            
            origin_ratio = ratio
            ratio = adjust_weight_a * r_mean_filtered + (1 - adjust_weight_a) * ratio
            
            mw = adjust_weight_w * w_mean_filtered + (1 - adjust_weight_w) * w
            if mw / ratio > 0.8:
                ratio = origin_ratio
        else: # based on llm advise size to perturb
            size_dict = {
                '1': (0.005, 0.0225),
                '2': (0.0225, 0.3 * 0.3),
                '3': (0.3 * 0.3, 0.5 * 0.5),
                '4': (0.5 * 0.5, 0.6 * 0.6),
                '5': (0.6 * 0.6, 0.7 * 0.8),
            }
            if 'size' in inst:
                try:
                    size = size_dict[str(round(float(inst['size'])))]
                except:
                    size = ''
                    for i in size_dict.keys():
                        if i in inst['size']:
                            size = size_dict[i]
                            break
                    if size=='':
                        size = size_dict['3']
            else:
                size = size_dict['3']

            if area < size[0] + (size[1] - size[0]) * 0.5:
                mw = w + np.random.uniform(0.0, w * preturb_ratio)
            else:
                mw = w - np.random.uniform(0.0, w * preturb_ratio)
            
            if ratio > 1.0:
                ratio = ratio * (1 + np.random.uniform(0.0, 0.3))
            else:
                ratio = ratio * (1 - np.random.uniform(0.0, 0.3))
        mh = mw / ratio
        
        new_box = find_best_direction(bbox, mw, mh, placed_boxes, shift_step)
        inst['bbox'] = new_box
    
    return basic_layout_info

def find_best_direction(box, perturbed_w, perturbed_h, placed_boxes, shift_step = 10):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = [cx - perturbed_w/2, cy - perturbed_h/2, cx + perturbed_w/2, cy + perturbed_h/2]

    best_box = box
    min_overlap = total_overlap(box, placed_boxes)
    
    left_dist, right_dist, top_dist, bottom_dist = distance_to_boundary(box)

    for i in range(shift_step):

        dx = random.uniform(-1 * left_dist, right_dist)
        dy = random.uniform(-1 * top_dist, bottom_dist)

        new_x = x1 + dx * 0.1
        new_y = y1 + dy * 0.1
        
        new_x = max(0, min(new_x, 1 - perturbed_w))
        new_y = max(0, min(new_y, 1 - perturbed_h))
        
        new_box = (new_x, new_y, new_x + perturbed_w, new_y + perturbed_h)
        overlap = total_overlap(new_box, placed_boxes)
        
        if overlap < min_overlap:  # 找到更小重叠的方向
            min_overlap = overlap
            best_box = new_box
    
    return best_box

def distance_to_boundary(box):
    x1, y1, x2, y2 = box
    left_dist = x1
    right_dist = 1 - x2
    top_dist = y1
    bottom_dist = 1 - y2
    return left_dist, right_dist, top_dist, bottom_dist

def overlap_area(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    overlap_x = max(0, min(x12, x22) - max(x11, x21))
    overlap_y = max(0, min(y12, y22) - max(y11, y21))
    return overlap_x * overlap_y

def total_overlap(box, placed_boxes):
    total_overlap_area = 0
    for placed_box in placed_boxes:
        if len(placed_box) == 4:
            total_overlap_area += overlap_area(box, placed_box)
    return total_overlap_area