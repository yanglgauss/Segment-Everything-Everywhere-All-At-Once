# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import os
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import torch
import argparse
import whisper
import numpy as np

from gradio import processing_utils
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

from demo.seem.tasks import *

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/focall_unicl_lang_demo.yaml", metavar="FILE", help='path to config file', )
    cfg = parser.parse_args()
    return cfg

'''
build args
'''
cfg = parse_option()
opt = load_opt_from_config_files([cfg.conf_files])
opt = init_distributed(opt)

# META DATA
cur_model = 'None'
if 'focalt' in cfg.conf_files:
    pretrained_pth = os.path.join("/data/DataSSD1/yli/SEEM/ckpt/seem_focalt_v0.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v0.pt"))
    cur_model = 'Focal-T'
elif 'focal' in cfg.conf_files:
    pretrained_pth = os.path.join("/data/DataSSD1/yli/SEEM/ckpt/seem_focall_v0.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt"))
    cur_model = 'Focal-L'

'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

'''
audio
'''
audio = whisper.load_model("base")

@torch.no_grad()
def inference(image, task, *args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        if 'Video' in task:
            return interactive_infer_video(model, audio, image, task, *args, **kwargs)
        else:
            return interactive_infer_image(model, audio, image, task, *args, **kwargs)

query_img = {}
query_img['image'] = PIL.Image.open("demo/seem/examples/zebras1.jpg")
query_img['mask'] = None

ref_img = {}
ref_img['image'] = PIL.Image.open("demo/seem/examples/zebras2.jpg")
ref_img['mask'] = None

task = ['Text']
reftxt = "zebra in the middle"

seg_img, caption = inference(query_img, task, refimg=ref_img, reftxt=reftxt)

# save seg_img as a jpg file
seg_img.save("./demo/seem/results/zebra_text.jpg")