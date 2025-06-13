# Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy

<a id="Installation"></a>
## Installation

## Conda environment setup
```
conda create -n IPCIR python=3.9 -y
conda activate IPCIR
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirement.txt
# Install MIGC
cd MIGC
pip install -e .
cd ..
cd Grounded-Segment-Anything
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
cd ..
cd GLIP
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
python setup.py build develop --user
cd ..
cd AutoGPTQ
pip install -vvv --no-build-isolation -e .
cd ..

pip install open_clip_torch
```

## Data preparation

### CIRCO

Download the CIRCO dataset following the instructions in the [**official repository**](https://github.com/miccunifi/CIRCO).

Alternatively, you can directly download the dataset (more straightforward):

- Download the images from [unlabeled2017.zip](http://images.cocodataset.org/zips/unlabeled2017.zip)

- Download the annotations from [image_info_unlabeled2017.zip](http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip)

After downloading the dataset, ensure that the folder structure matches the following:

```
├── CIRCO
│   ├── annotations
|   |   ├── [test | test_multi_gpt3-5].json

│   ├── COCO2017_unlabeled
|   |   ├── annotations
|   |   |   ├──  image_info_unlabeled2017.json
|   |   ├── unlabeled2017
|   |   |   ├── [000000243611.jpg | 000000535009.jpg | ...]
```

### CIRR

Download the CIRR dataset following the instructions in the [**official repository**](https://github.com/Cuberick-Orion/CIRR).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── CIRR
│   ├── train
|   |   ├── [0 | 1 | 2 | ...]
|   |   |   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]

│   ├── dev
|   |   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]

│   ├── test1
|   |   ├── [test1-0-0-img0.png | test1-0-0-img1.png | ...]

│   ├── cirr
|   |   ├── captions
|   |   |   ├── cap.rc2.[train | val | test1].json
|   |   ├── image_splits
|   |   |   ├── split.rc2.[train | val | test1].json
```

### FashionIQ

Download the FashionIQ dataset following the instructions in the [**official repository**](https://github.com/XiaoxiaoGuo/fashion-iq).

After downloading the dataset, ensure that the folder structure matches the following:

```
├── FashionIQ
│   ├── captions
|   |   ├── cap.dress.[train | val | test].json
|   |   ├── cap.toptee.[train | val | test].json
|   |   ├── cap.shirt.[train | val | test].json

│   ├── image_splits
|   |   ├── split.dress.[train | val | test].json
|   |   ├── split.toptee.[train | val | test].json
|   |   ├── split.shirt.[train | val | test].json

│   ├── images
|   |   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]
```
## Model preparation
Our method requires you to download some model weights in advance. The details are as follows:

### Layout Generation
IP-CIR uses Qwen to infer the model layout, so you need to download the Qwen model from the following path:
[Qwen/Qwen1.5-32B-Chat](https://huggingface.co/Qwen/Qwen1.5-32B-Chat)
Our work primarily runs on [Qwen1.5-32B-Chat-GPTQ-Int4](https://huggingface.co/Qwen/Qwen1.5-32B-Chat-GPTQ-Int4), which requires less GPU memory but requires prior installation of AutoGPTQ.
Additionally, in theory, other Qwen models or LLMs like OpenAI's GPT are also feasible.


### Image Generation
To generate Proxy Images, you need to download the following weights:
Layout Controller: [MIGC](https://drive.google.com/file/d/107fnQ9Fpu5K0UtqnHlKja7hqe-GwjAmz/view?usp=sharing)
Image Generator:  [RV](aishuu/Realistic_Vision_V6.0_B1VAE)
[ELITE](https://github.com/csyxwei/ELITE), (only the global and local mapper weights are needed)
[Grounding-dino](https://huggingface.co/ShilongLiu/GroundingDINO/blob/main/groundingdino_swint_ogc.pth)
[SAM](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth)
[GLIP](https://huggingface.co/GLIPModel/GLIP/blob/main/glip_large_model.pth)
[CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)
[CLIP-G](https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s12B-b42K)
Then, create a weights folder in your project directory and organize the downloaded weights as follows:
```
/Imagine-and-Seek/weights
├── CLIP-ViT-g-14-laion2B-s12B-b42K
│   ├── config.json
│   ├── merges.txt
│   ├── open_clip_config.json
│   ├── open_clip_pytorch_model.bin
│   ├── preprocessor_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── glip_large_model.pth
├── global_mapper.pt
├── groundingdino_swint_ogc.pth
├── local_mapper.pt
├── MIGC_SD14.ckpt
├── realisticVisionV60B1_v60B1VAE.safetensors
├── sam_vit_h_4b8939.pth
└── ViT-L-14.pt
```

## Generate Layout for Proxy Image
Layout generation is the first step of IP-CIR. For the images and relative captions contained in a single query, it involves envisioning a reasonable layout for the final image. In this step, we assume you have already completed the preliminary steps of LDRE (1. Basic Step of LDRE below) and obtained the captions for each image in the dataset as well as the captions conceived by GPT. Based on this data, you can use the following command to synthesize your desired layout.
```sh
sh layout_generation.sh
```
You need to first complete the corresponding paths in the running script, and then set the dataset and test split you want to process by changing the mode.

## Generate Proxy Image
Next, you need to generate the Proxy Image. It's important to note that what we provide here is just one optional approach for generating Proxy Images—specifically, a controllable generation scheme using layout + MIGC.

You may also skip the layout generation phase and instead use image editing tools to create the Proxy Image, or directly employ T2I models like FLUX for generation.

If you'd like to follow our generation steps, you can use the following command to perform the generation:
```sh
sh proxy_generation.sh
```
You need to first complete the corresponding paths in the running script.

## Retrieval with proxy image
We have placed the datasets we generated and processed, along with the proxy images, in [GOOGLE](https://drive.google.com/file/d/1KRPT715i8my5bHAOKflhqS4_gbt0s3YU/view?usp=drive_link). You can download and use them, or follow the steps below to generate your own proxy images. You can also try building or implementing your own proxy construction or retrieval solution.

After downloading and extracting the files, the directory structure is as follows:
```
circo_test
├── images
│   ├── aug_features_G.npz
│   ├── aug_features_L.npz
│   ├── combined
│   ├── source_features_G.npz
│   └── source_features_L.npz
|   └── ipcir_layout.json
└── test.json
```
The images folder contains our synthesized proxy images and cached CLIP features, while ipcir_layout.json stores the layouts we synthesized using Qwen. The test.json file contains the text information processed according to the LDRE steps.

When using them, you need to replace the original CIRCO dataset annotations with test.json. Additionally, in retrieval.sh, modify the aug_dir and layout_path to the corresponding paths:

aug_dir should point to the path of the circo_test/images folder.

layout_path should point to the path of ipcir_layout.json.

### 1. Basic Step of LDRE
First, the basic preparatory work required for the LDRE task needs to be completed.

Run the following command to generate dense captions:

```sh
python src/dense_caption_generator.py
python src/reasoning_and_editing.py
```
You can refer to LDRE github for more details (such as environment). 

#### 2. IP-CIR

To generate the predictions file to be uploaded on the [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/) or on the [CIRCO Evaluation Server](https://circo.micc.unifi.it/) run the following command:

```sh
python src/divergent_caption_ensemble.py
```

The predictions file will be saved in the `data/test_submissions/{dataset}/` folder.

We have provided the experimental results of our LDRE for your evaluation on the [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/) or on the [CIRCO Evaluation Server](https://circo.micc.unifi.it/), in the `data/test_submissions/{dataset}/` folder.



## 🏫About us
Thank you for your interest in this project. The project is supervised by the ReLER Lab at Zhejiang University’s College of Computer Science and Technology and [HUAWEI](https://www.huawei.com/cn/). ReLER was established by [Yang Yi](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en), a Qiu Shi Distinguished Professor at Zhejiang University. Our dedicated team of contributors includes [You Li](https://scholar.google.com/citations?user=2lRNus0AAAAJ&hl=en&oi=sra),[Fan Ma](https://scholar.google.com/citations?hl=en&user=FyglsaAAAAAJ), [Yi Yang](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en).

## Contact us
If you have any questions, feel free to contact me via email zdw1999@zju.edu.cn 

## Acknowledgements
Our work is based on [stable diffusion](https://github.com/Stability-AI/StableDiffusion), [diffusers](https://github.com/huggingface/diffusers), [MIGC](https://github.com/limuloo/MIGC) and [LDRE](https://github.com/yzy-bupt/LDRE). We appreciate their outstanding contributions.

## Citation
If you find this repository useful, please use the following BibTeX entry for citation.
```

@InProceedings{Li_2025_CVPR,
    author    = {Li, You and Ma, Fan and Yang, Yi},
    title     = {Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {3984-3993}
}
```
