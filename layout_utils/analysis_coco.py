import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# 定义数据路径
dataDir = '/mnt/data0/liyou/data/COCO2017'
dataType = 'train2017'
annFile = f'{dataDir}/annotations/instances_{dataType}.json'

# 加载COCO数据集
coco = COCO(annFile)

# 获取所有类别的ID和名称
catIds = coco.getCatIds()
cats = coco.loadCats(catIds)
catNames = {cat['id']: cat['name'] for cat in cats}

# 创建一个字典来存储每个类别的统计信息
category_stats = {catNames[catId]: {'length': [], 'aspect_ratio': [], 'center_x': [], 'center_y': []} for catId in catIds}

# 遍历所有图像ID
imgIds = coco.getImgIds()
for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    
    for ann in anns:
        # 过滤掉面积小于图像面积十二分之一的物体
        if ann['iscrowd'] == 1 or ann['bbox'][3] == 0.0:
            continue
        
        # 获取类别名称
        catName = catNames[ann['category_id']]
        
        # 获取边界框信息
        bbox = ann['bbox']
        length = bbox[2] / img['width']  # 归一化长度
        aspect_ratio = bbox[2] / bbox[3]  # 长宽比
        center_x = (bbox[0] + bbox[2] / 2) / img['width']  # 归一化中心点x坐标
        center_y = (bbox[1] + bbox[3] / 2) / img['height']  # 归一化中心点y坐标
        
        # 存储统计信息
        category_stats[catName]['length'].append(length)
        category_stats[catName]['aspect_ratio'].append(aspect_ratio)
        category_stats[catName]['center_x'].append(center_x)
        category_stats[catName]['center_y'].append(center_y)

# 计算每个类别的均值和标准差
category_distributions = {}
for catName in category_stats:
    if category_stats[catName]['length']:  # 确保类别中有数据
        length_mean, length_std = np.mean(category_stats[catName]['length']), np.std(category_stats[catName]['length'])
        aspect_ratio_mean, aspect_ratio_std = np.mean(category_stats[catName]['aspect_ratio']), np.std(category_stats[catName]['aspect_ratio'])
        center_x_mean, center_x_std = np.mean(category_stats[catName]['center_x']), np.std(category_stats[catName]['center_x'])
        center_y_mean, center_y_std = np.mean(category_stats[catName]['center_y']), np.std(category_stats[catName]['center_y'])
        
        category_distributions[catName] = {
            'length_mean': length_mean,
            'length_std': length_std,
            'aspect_ratio_mean': aspect_ratio_mean,
            'aspect_ratio_std': aspect_ratio_std,
            'center_x_mean': center_x_mean,
            'center_x_std': center_x_std,
            'center_y_mean': center_y_mean,
            'center_y_std': center_y_std
        }

import json
with open('/home/liyou/code/MMAug/mmaug_tool/coco_stastic.json','w') as f:
    json.dump(category_distributions, f)

# 打印每个类别的统计结果
for catName, stats in category_distributions.items():
    print(f"Category: {catName}")
    print("  Length: Mean =", stats['length_mean'], ", Std =", stats['length_std'])
    print("  Aspect Ratio: Mean =", stats['aspect_ratio_mean'], ", Std =", stats['aspect_ratio_std'])
    print("  Center X: Mean =", stats['center_x_mean'], ", Std =", stats['center_x_std'])
    print("  Center Y: Mean =", stats['center_y_mean'], ", Std =", stats['center_y_std'])
    print()

# 绘制正态分布曲线
def plot_normal_distribution(mean, std, title, xlabel):
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.show()

# 绘制每个类别的正态分布曲线
for catName, stats in category_distributions.items():
    plot_normal_distribution(stats['length_mean'], stats['length_std'], f"{catName} - Length Distribution", 'Length')
    plot_normal_distribution(stats['aspect_ratio_mean'], stats['aspect_ratio_std'], f"{catName} - Aspect Ratio Distribution", 'Aspect Ratio')
    plot_normal_distribution(stats['center_x_mean'], stats['center_x_std'], f"{catName} - Center X Distribution", 'Center X Coordinate')
    plot_normal_distribution(stats['center_y_mean'], stats['center_y_std'], f"{catName} - Center Y Distribution", 'Center Y Coordinate')

# 采样函数
def sample_from_normal_distribution(mean, std, size=1):
    return np.random.normal(mean, std, size)

# 示例：从某个类别的分布中采样10个样本（以"person"为例）
if 'person' in category_distributions:
    sampled_lengths = sample_from_normal_distribution(category_distributions['person']['length_mean'], category_distributions['person']['length_std'], 10)
    sampled_aspect_ratios = sample_from_normal_distribution(category_distributions['person']['aspect_ratio_mean'], category_distributions['person']['aspect_ratio_std'], 10)
    sampled_center_x = sample_from_normal_distribution(category_distributions['person']['center_x_mean'], category_distributions['person']['center_x_std'], 10)
    sampled_center_y = sample_from_normal_distribution(category_distributions['person']['center_y_mean'], category_distributions['person']['center_y_std'], 10)

    print("Sampled Lengths:", sampled_lengths)
    print("Sampled Aspect Ratios:", sampled_aspect_ratios)
    print("Sampled Center X:", sampled_center_x)
    print("Sampled Center Y:", sampled_center_y)
