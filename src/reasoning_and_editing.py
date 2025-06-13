import json
import time
import traceback

from tqdm import tqdm
# import openai
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse

parser = argparse.ArgumentParser(description="Generate a question-answer instruction tuning dataset.")
parser.add_argument('--dress',default='toptee',type=str)
parser.add_argument('--model_path',default='/mnt/data0/liyou/ckpt/Qwen1.5-32B-Chat-GPTQ-Int4',type=str)
args = parser.parse_args()

# openai.api_key = "your_api_key"
Qwen_model_path = args.model_path

tokenizer = AutoTokenizer.from_pretrained(Qwen_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(Qwen_model_path, device_map="auto", trust_remote_code=True).eval()

# DATASET = 'circo' # 'cirr', 'fashioniq'
DATASET = 'cirr_val'

if DATASET == 'circo':
    SPLIT = 'test'
    input_json = 'CIRCO/annotations/test.json'
elif DATASET == 'circo_val':
    SPLIT = 'val'
    input_json = 'CIRCO/annotations/val.json'
elif DATASET == 'cirr':
    SPLIT = 'test1'
    input_json = 'CIRR/cirr/captions/cap.rc2.test1.json'
elif DATASET == 'cirr_val':
    SPLIT = 'val'
    input_json = 'CIRR/cirr/captions/cap.rc2.val.json'
elif DATASET == 'fashioniq':
    SPLIT = 'val'
    DRESS = args.dress # 'shirt', 'toptee', 'dress
    new_annotations = []
    input_json = f'/mnt/data1/comp_data/fashion-iq/captions/cap.{DRESS}.{SPLIT}.json'

BLIP2_MODEL = 'opt' # or 'opt' or 't5'
MULTI_CAPTION = True
NUM_CAPTION = 15
with open(input_json, "r") as f:
    annotations = json.load(f)

for ans in tqdm(annotations):
    if DATASET == 'circo':
        rel_cap = ans["relative_caption"]
    elif DATASET == 'circo_val':
        rel_cap = ans["relative_caption"]
    elif DATASET == 'cirr':
        rel_cap = ans["caption"]
    elif DATASET == 'cirr_val':
        rel_cap = ans["caption"]
    else:
        rel_cap = ans['captions'][0]

    if MULTI_CAPTION:
        blip2_caption = ans["multi_caption_{}".format(BLIP2_MODEL)]
    else:
        if BLIP2_MODEL == 'none':
            blip2_caption = ans["shared_concept"]
        elif BLIP2_MODEL == 'opt':
            blip2_caption = ans["blip2_caption"]
        else:
            blip2_caption = ans["blip2_caption_{}".format(BLIP2_MODEL)]

    sys_prompt = """
        I have an image. Given an instruction to edit the image, carefully generate a description of the edited image. I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". 
        The edited description you generate should begin with \"Edited Description:\". "
        You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.
    """
    if MULTI_CAPTION:
        multi_gpt = []
        for cap in blip2_caption:
            # print(prompt)
            while True:
                try:
                    messages = [
                        {"role": "system", "content": sys_prompt}
                    ]
                    messages.append({"role": "user", "content": "Image Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\n"})
                    messages.append({"role": "assistant", "content": "Edited Description: a woman adjusting a man's tie."})
                    messages.append({"role": "user", "content": "Image Content: {}\nInstruction: {}\nEdited Description:".format(cap, rel_cap)})
                    device = "cuda" 
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    model_inputs = tokenizer([text], return_tensors="pt").to(device)

                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=1024
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    multi_gpt.append(response.strip('\n'))
                    break
                except:
                    traceback.print_exc()
                    time.sleep(3)
        ans["multi_gpt-3.5_{}".format(BLIP2_MODEL)] = multi_gpt
        
    else:
        usr_prompt = "I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". The edited description you generate should begin with \"Edited Description:\". You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.\nA example:\nImage Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\nEdited Description: a woman adjusting a man's tie.\n\nImage Content: {}\nInstruction: {}\nEdited Description:".format(blip2_caption, rel_cap)
        # print(prompt)
        while True:
            try:
                messages=[{"role": "system",
                            "content": sys_prompt},
                            {"role": "user", "content": usr_prompt}],
                device = "cuda" 
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(device)

                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_token
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                if BLIP2_MODEL == 'opt':
                    ans["gpt-3.5-turbo"] = responses.strip('\n')
                else:
                    ans["gpt-3.5-turbo_{}".format(BLIP2_MODEL)] = responses.strip('\n')
                break
            except:
                traceback.print_exc()
                time.sleep(3)


with open(input_json, "w") as f:
    f.write(json.dumps(annotations, indent=4))