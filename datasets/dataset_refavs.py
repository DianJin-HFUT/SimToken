import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pdb

import sys
import os
import random

from torchvision import transforms
from collections import defaultdict
import cv2

# cv2.setNumThreads(0)

import transformers
from transformers import AutoImageProcessor, AutoTokenizer, AutoModel
from PIL import Image

from towhee import pipe, ops
from transformers import pipeline
from transformers import CLIPImageProcessor
from models.segment_anything.utils.transforms import ResizeLongestSide

# logger = log_agent('audio_recs.log')

import pickle as pkl
from models.llava import conversation as conversation_lib
from collections import defaultdict


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VIDEO_TOKEN = "<video>"

AUDIO_TOKEN_INDEX = -300
DEFAULT_AUDIO_TOKEN = "<audio>"

class REFAVS(Dataset):
    def __init__(self, split='train', cfg=None, tokenizer=None, input_type='video'):
        self.input_type = input_type
        self.data_dir = cfg.data_dir

        # 读metadata 并将split，比如属于train的数据区分出来
        meta_path = f'{self.data_dir}/metadata_a.csv'


        metadata = pd.read_csv(meta_path, header=0)
        self.split = split
        self.metadata = metadata[metadata['split'] == split]  # split= train,test,val.

        # 构建一个初始元素为空list的字典
        self.video_to_samples = defaultdict(list)
        self.audio_to_captions = defaultdict(list)
        # 存储所有的uid，对应独特的视频，list中存储的是在metadata中的位置
        for i in range(len(self.metadata)):
            row = self.metadata.iloc[i]
            vid = row['uid'].rsplit('_', 2)[0]
            self.video_to_samples[vid].append(i)
            # if split == 'test_n':
            #     self.audio_to_captions[vid] = 'silent audio'
            # else:
            #     self.audio_to_captions[vid] = row['desc_a']

        self.all_vids = list(self.video_to_samples.keys())
        # print("all_vids", len(self.all_vids))



        self.media_path = f'{self.data_dir}/media'
        self.label_path = f'{self.data_dir}/gt_mask'
        self.frame_num = cfg.frame_n #10
        self.text_max_len = cfg.text_max_len

        self.tokenizer = tokenizer

        # 对话模板
        if cfg.conv_template == 0:
            self.system = "\nReference Video: <video> \nTarget Image: <image> \n"
        elif cfg.conv_template == 1:
            self.system = "\nReference Video: <video> \nReference Audio: <audio> \nTarget Image: <image> \n"
        elif cfg.conv_template == 2:
            self.system = "\nReference Video: <video> \nAudio description: {audio_des} \nTarget Image: <image> \n"
        elif cfg.conv_template == 3:
            self.system = "\nReference Video: <video> \nReference Audio: <audio> \nAudio description: {audio_des} \nTarget Image: <image> \n"
        elif cfg.conv_template == 4:
            self.system = "\n"
        elif cfg.conv_template == 5:
            self.system = "Reference Audio: <audio> \nTarget Image: <image> \n"

        self.question = "What is {sent} in the Reference Video? Please respond with segmentation mask in the Target Image."



        # 用于输入大模型的视觉处理
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(cfg.vision_tower)

        # 用于输入samdecoder的视觉处理
        self.transform = ResizeLongestSide(1024)

        self.pixel_mean = torch.Tensor([113.263, 99.370, 92.492]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([64.274, 61.068, 58.626]).view(-1, 1, 1)
        self.img_size = 1024
        # self.img_size = 224

    # 处理 <image>
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x



    def __len__(self):
        if self.input_type == 'refer':
            return len(self.metadata)
        elif self.input_type == 'video':
            return len(self.all_vids)

    def __getitem__(self, idx):

        if self.input_type == 'refer' :
            vid = self.metadata.iloc[idx]['uid'].rsplit('_', 2)[0]
            indices = [idx]
        elif self.input_type == 'video':
            # 此处idx不再直接从metadata中索引，而是去all_vids中索引独特的vid，对应独特的视频
            vid = self.all_vids[idx]
            # indices是一个list 其中是当前vid对应的视频，在metadata中是哪些行
            indices = self.video_to_samples[vid]  # 获取所有该视频对应的行


        # 统一资源加载（video embedding、audio、frames 等）
        feat_aud = torch.load(f'/home/u2024110507/Ref_AVS/data/audio_embed/{vid}.pt')
        image_feat = torch.load(f'{self.data_dir}/image_embed/{vid}.pt')

        img_clips = []
        masks = []
        images = []
        rec_texts = []
        target_ids = []

        conversations = []

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        # print("conv.system:", conv.system)

        conv.system += self.system.format(audio_des=self.audio_to_captions[vid])

        # 根据indices去metadata中读对应的行
        for i, meta_idx in enumerate(indices):
            row = self.metadata.iloc[meta_idx]
            refer = row['exp'].lower().rstrip('.')
            fid = row['fid']  # 当前 refer 对应目标 ID

            conv.append_message(conv.roles[0], self.question.format(sent=refer))
            conv.append_message(conv.roles[1], "Sure, it is [SEG]")

            rec_texts.append(refer)
            target_ids.append(fid)

            temp_mask = []
            for frame_idx in range(self.frame_num):
                path_mask = f'{self.label_path}/{vid}/fid_{fid}/0000{frame_idx}.png'
                mask_cv2 = cv2.imread(path_mask)
                mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)
                gt_binary_mask = torch.as_tensor(mask_cv2 > 0, dtype=torch.float32)
                temp_mask.append(gt_binary_mask)
            masks.append(torch.stack(temp_mask, dim=0)) # list[num_refer] :[10, 3, H, W]

        orgsize = masks[0].shape[-2:]

        conversation = conv.get_prompt()

        for _idx in range(self.frame_num):
            path_frame = f'{self.media_path}/{vid}/frames/{_idx}.jpg'
            image = cv2.imread(path_frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_clip = self.clip_image_processor(image, return_tensors="pt")["pixel_values"][0]  # [3, 224, 224]
            image = self.transform.apply_image(image)
            resize = image.shape[:2]
            image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())  # [3, 1024, 1024]

            images.append(image)
            img_clips.append(image_clip)




        images = torch.stack(images, dim=0)  # [10, 3, 1024, 1024]
        img_clips = torch.stack(img_clips, dim=0)  # [10, 3, 224, 224]
        masks = torch.stack(masks, dim=0)  # [num_refer, 10, 3, H, W]

        return {
            'vid': vid,
            'image': images,  # [10, 3, 1024, 1024]
            'img_clip': img_clips,  # [10, 3, 224, 224]
            'mask': masks,  # [num_refer, 10, 3, H, W]
            'conversation': conversation,
            'feat_aud': feat_aud,  # [10, 128]
            'resize': resize,
            'orgsize': orgsize,
            'feat_sam': image_feat,    # [T, 256, 64, 64]
            'ref': rec_texts,
            'fids': target_ids,
        }


