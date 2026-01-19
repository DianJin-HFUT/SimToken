import transformers
from datasets import REFAVS
from configs import args
from torch.utils.data import DataLoader
from functools import partial
from models.llava import conversation as conversation_lib
# from  models.avs_model import VISAForCausalLM
from  models.avs_model import Simtoken_ForCausalLM

import torch
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
from torch import optim
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from utils import utility
import random
import numpy as np
import re
import time
import os


import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VIDEO_TOKEN = "<video>"

AUDIO_TOKEN_INDEX = -300
DEFAULT_AUDIO_TOKEN = "<audio>"

def set_seed(seed: int = 42):
    """
    完全固定 PyTorch 训练的所有随机性。
    适用于 CPU、GPU、Dataloader 多线程环境。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)  # 保证 python 内部 hash 顺序一致
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA 11+，CUBLAS 确保确定性

    torch.backends.cudnn.deterministic = True  # 禁用非确定性算法
    torch.backends.cudnn.benchmark = False     # 禁用自动优化算法选择（它可能非确定性）
    # torch.use_deterministic_algorithms(True)   # 强制 PyTorch 报错任何非确定性操作

def seed_worker(worker_id):
    """用于 DataLoader 内部 worker 线程的随机数固定"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
                isinstance(input_dict[k], list)
                and len(input_dict[k]) > 0
                and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict

def tokenizer_image_audio_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX, num_frames=10, return_tensors=None):
    # 使用正则表达式分割 prompt，同时识别 <image> 和 <audio>
    prompt_chunks = re.split(r'(<image>|<audio>|<video>)', prompt)

    # 过滤掉空字符串
    prompt_chunks = [chunk for chunk in prompt_chunks if chunk]

    # 将 prompt_chunks 分成两类：文本部分和 token 部分
    text_chunks = []
    token_types = []
    for chunk in prompt_chunks:
        if chunk == "<image>":
            token_types.append("image")
        elif chunk == "<audio>":
            token_types.append("audio")
        elif chunk == "<video>":
            token_types.append("video")
        else:
            text_chunks.append(chunk)

    # Tokenize 文本部分
    tokenized_chunks = [tokenizer(chunk).input_ids for chunk in text_chunks]

    def insert_separators(text_chunks, tokenized_chunks, token_types, image_token_index, audio_token_index, num_frames):
        input_ids = []
        offset = 0
        if (
                len(tokenized_chunks) > 0
                and len(tokenized_chunks[0]) > 0
                and tokenized_chunks[0][0] == tokenizer.bos_token_id
        ):
            offset = 1
            input_ids.append(tokenized_chunks[0][0])

        # 确保 text_chunks 和 token_types 的长度一致
        min_length = min(len(text_chunks), len(token_types))
        for i in range(min_length):
            # 添加文本部分
            input_ids.extend(tokenized_chunks[i][offset:])
            # 添加对应的 token index
            if token_types[i] == "image":
                input_ids.append(image_token_index)
            elif token_types[i] == "audio":
                input_ids.append(audio_token_index)
            elif token_types[i] == "video":
                input_ids.extend([image_token_index] * num_frames)

        # 如果 text_chunks 比 token_types 长，添加剩余的文本部分
        if len(text_chunks) > min_length:
            input_ids.extend(tokenized_chunks[min_length][offset:])

        return input_ids

    input_ids = insert_separators(text_chunks, tokenized_chunks, token_types, image_token_index, audio_token_index, num_frames)

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids

def collate_fn(batch, tokenizer=None):
    vids = []
    images = []
    image_clips = []
    masks = []
    conversations = []
    audio_feats = []
    image_feats = []
    resizes = []
    orgsizes = []
    refs = []
    refs_num = []
    fids = []


    for data in batch:
        vids.append(data['vid'])
        images.append(data['image'])
        image_clips.append(data['img_clip'])
        masks.append(data['mask'])
        conversations.append(data['conversation'])
        audio_feats.append(data['feat_aud'])
        resizes.append(data['resize'])
        orgsizes.append(data['orgsize'])
        image_feats.append(data['feat_sam'])
        refs_num.append(len(data['ref']))
        fids.append(data['fids'])

        refs.append(data['ref'][0])


    # input_ids = [tokenizer_image_token(conv, tokenizer, return_tensors="pt") for conv in conversations]
    input_ids = [tokenizer_image_audio_token(conv, tokenizer, return_tensors="pt") for conv in conversations]  # list
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    ref_ids = [tokenizer_image_audio_token(ref, tokenizer, return_tensors="pt") for ref in refs]

    conv = conversation_lib.default_conversation.copy()
    labels = input_ids.clone()

    # sep = conv.sep + conv.roles[1] + ": " # “###Assistant：”
    sep = 'Sure, it is [SEG]'

    for conversation, target in zip(conversations, labels):
        # 用sep分割用户和系统
        parts = conversation.split(sep)
        # print(parts)

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        # Sure, it is [seg] 占的id数量
        sep_len = len(tokenizer_image_audio_token(sep, tokenizer)) - 1


        for i in range(len(parts)-1):
            part_len = len(tokenizer_image_audio_token(parts[i], tokenizer)) - 2
            target[cur_len: cur_len + part_len] = IGNORE_INDEX
            cur_len += part_len + sep_len

        target[cur_len:] = IGNORE_INDEX


    return {"vids": vids,
            "images": images,  # list[B]:[T, 3, 1024, 1024]
            "images_clip": image_clips,  # list[B]:[T, 3, 224, 224]
            "masks": masks,  # list[B]:[num_ref, T, H, W]
            "convs": conversations,  # list[B]: str
            "input_ids": input_ids,  # list[B]:[max_len]
            "attention_masks": attention_masks,  # list[B]:[max_len]
            "labels": labels,  # list[B]:[max_len]
            "audio_feats": audio_feats,  # list[B]:[10, 128]
            "resizes": resizes,  # list[B]
            "orgsizes": orgsizes,  # list[B]
            "image_feats": image_feats,
            "ref_ids": ref_ids,  # list[B]: [ref_id_len]
            "refs_num": refs_num,
            "fids": fids
    }


import torch.multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn")
    set_seed(42)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/home/u2024110507/Ref_AVS/models/ChatUnivi7B",  # path/to/Chatunivi 从sh文件来看其也是手动下载下来指定的
        cache_dir=None,
        model_max_length=2048,  # 2048
        padding_side="right",
        use_fast=False,
    )
    # 在tokennizer中加入segtoken等
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]  # 32000
    print("seg_token_idx: ", seg_token_idx)

    train_dataset = REFAVS('train', args, tokenizer, input_type='refer')
    val_dataset_s_refer = REFAVS('test_s', args, tokenizer, input_type='refer')
    val_dataset_u_refer = REFAVS('test_u', args, tokenizer, input_type='refer')
    val_dataset_n_refer = REFAVS('test_n', args, tokenizer, input_type='refer')


    g = torch.Generator()
    g.manual_seed(42)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, worker_init_fn=seed_worker,collate_fn=partial(collate_fn, tokenizer=tokenizer), generator=g)

    val_dataloader_s_refer = DataLoader(val_dataset_s_refer, batch_size=4, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_dataloader_u_refer = DataLoader(val_dataset_u_refer, batch_size=4, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_dataloader_n_refer = DataLoader(val_dataset_n_refer, batch_size=4, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))


    model_args = {
        "train_mask_decoder": True,  # store_true 所以应该是true or false 应该是用来决定是否要训练decoder
        "out_dim": 256,  # 256
        "ce_loss_weight": 1.0,
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "seg_token_idx": seg_token_idx,  # [SEG] 标记的索引（input_ids）
        "vision_pretrained": args.vision_pretrained,  # sam_vit_h_xxx.pth
        "vision_tower": args.vision_tower,
        "use_im_start_end": False,
        "compress": args.compress,
        "start": args.start,
    }
    # model = VISAForCausalLM.from_pretrained(args.mllm, torch_dtype=torch.float32, low_cpu_mem_usage=True, **model_args)
    model = Simtoken_ForCausalLM.from_pretrained(args.mllm, torch_dtype=torch.float32, low_cpu_mem_usage=True, **model_args)
    print("\nmodel loaded")

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.float32, device="cuda")

    model_args_from_pt = AutoConfig.from_pretrained(args.mllm)
    model_args_from_pt.use_cluster = True
    model_args_from_pt.freeze = False
    model_args_from_pt.mm_tune = True
    model_args_from_pt.spatial_cluster_rate0 = 64
    model_args_from_pt.spatial_cluster_rate1 = 32
    model_args_from_pt.spatial_cluster_rate2 = 16
    model_args_from_pt.temporal_cluster_rate = 0.0625
    model_args_from_pt.use_cluster = True
    model_args_from_pt.vision_tune = False
    model.get_model().initialize_cluster_modules(model_args_from_pt)

    model.get_model().initialize_lisa_modules(model.get_model().config)
    # 冻结视觉模块（vision_tower）和多模态投影器
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    lora_r = 8
    target_modules = "q_proj,v_proj"
    if lora_r > 0:
        # 定义一个函数 find_linear_layers，用于查找模型中需要应用 LoRA 的线性层
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            # 遍历模型的所有模块
            for name, module in model.named_modules():
                if (
                        # 检查模块是否是线性层
                        isinstance(module, cls)
                        # 排除某些特定模块
                        and all(
                    [
                        x not in name
                        for x in [
                        "visual_model",
                        "vision_tower",
                        "mm_projector",
                        "text_hidden_fcs",
                        "audio_feature_layer",
                    ]
                    ]
                )
                        # 检查模块名称是否包含目标模块名称（lora_target_modules）
                        and any([x in name for x in lora_target_modules])
                ):
                    # 将符合条件的模块名称添加到 lora_module_names 集合中
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))


        lora_alpha = 16
        lora_dropout = 0.05
        # args.lora_target_modules默认是 q_proj,v_proj
        # 找到需要lora的层
        lora_target_modules = find_linear_layers(
            model, target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # peft 库中的一个函数，用于将 LoRA 配置（lora_config）应用到模型（model）中。为每个目标模块添加低秩矩阵（LoRA 层）。返回一个经过 LoRA 微调的新模型实例。
        model = get_peft_model(model, lora_config)
        print("\nLora deployed")

        model.print_trainable_parameters()

    model = model.to("cuda")
    model.resize_token_embeddings(len(tokenizer))


    for name, param in model.audio_feature_layer.named_parameters():
        param.requires_grad = True
        # print(name, param.requires_grad)
    # for name, param in model.token_compressor.named_parameters():
    #     param.requires_grad = True


    for n, p in model.named_parameters():
        if any(
                [
                    x in n
                    for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
                ]
        ):
            p.requires_grad = True


    print("will save train model")

    def valuate(model, dataloader, args, name):
        model.eval()

        total_iou = 0
        total_fscore = 0
        count = 0

        for batch in tqdm(dataloader, desc=f"Evaluating on {name}"):
            input_dict = dict_to_cuda(batch)
            with torch.no_grad():
                output_dict = model.forward(images=input_dict["images"],
                                            images_clip=input_dict["images_clip"],
                                            audio_features=input_dict["audio_feats"],
                                            image_features=input_dict["image_feats"],
                                            input_ids=input_dict["input_ids"],
                                            labels=input_dict["labels"],
                                            attention_masks=input_dict["attention_masks"],
                                            masks_list=input_dict["masks"],
                                            resize_list=input_dict["resizes"],
                                            orgsize_list=input_dict["orgsizes"],
                                            conversation_list=input_dict["convs"],
                                            refs_num=input_dict["refs_num"],
                                            fids=input_dict["fids"],
                                            vids=input_dict["vids"],
                                            contrast=args.ct_weight,
                                            ref_ids=input_dict["ref_ids"],
                                            inference=True)
            pred_masks = output_dict["pred_masks"]  # list[B]:[num_seg, T, H, W]
            gt_masks = output_dict["gt_masks"]  # list[B]:[num_seg, T, H, W]
            for i in range(len(pred_masks)):
                num_seg = pred_masks[i].shape[0]
                T = pred_masks[i].shape[1]
                iou = utility.mask_iou(pred_masks[i], gt_masks[i])
                fscore = utility.Eval_Fmeasure(pred_masks[i], gt_masks[i], None)

                total_iou += iou * num_seg * T
                total_fscore += fscore * num_seg * T
                count += num_seg * T

        print(f"\n  valuate on {name}:  miou: {total_iou/count}  fscore: {total_fscore/count}")

        with open(f"/home/u2024110507/Ref_AVS/checkpoints/{args.name}.txt", "a") as f:
            f.write(f"valuate on {name}:  miou {total_iou/count}  true fscore {total_fscore/count} \n")


    # ---------------训练------------------------------------------

    model.train()
    epochs = args.epochs
    print("init lr:", args.lr)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    gradient_accumulation_steps = int(16 // args.batch_size)
    step_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    total_steps = epochs * step_per_epoch
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        loop = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(loop):
            input_dict = dict_to_cuda(batch)
            output_dict = model.forward(images=input_dict["images"],
                                        images_clip=input_dict["images_clip"],
                                        audio_features=input_dict["audio_feats"],
                                        image_features=input_dict["image_feats"],
                                        input_ids=input_dict["input_ids"],
                                        labels=input_dict["labels"],
                                        attention_masks=input_dict["attention_masks"],
                                        masks_list=input_dict["masks"],
                                        resize_list=input_dict["resizes"],
                                        orgsize_list=input_dict["orgsizes"],
                                        conversation_list=input_dict["convs"],
                                        refs_num=input_dict["refs_num"],
                                        fids=input_dict["fids"],
                                        vids=input_dict["vids"],
                                        contrast=args.ct_weight,
                                        ref_ids=input_dict["ref_ids"],
                                        epoch=epoch,
                                        inference=False)

            loss = output_dict["loss"]
            loss = loss / gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()


            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                current_lr = scheduler.get_lr()[0]
                loop.set_postfix(lr=current_lr, loss=running_loss / ((step + 1) / gradient_accumulation_steps))

        print(f"  Epoch {epoch + 1}, Loss:{running_loss / ((step + 1) / gradient_accumulation_steps) :.4f}, Learning Rate:{scheduler.get_last_lr()[0]:.6f}")


        with open(f"/home/u2024110507/Ref_AVS/checkpoints/{args.name}.txt", "a") as f:
            f.write(f"Epoch {epoch}: running_loss {running_loss / len(train_dataloader) * gradient_accumulation_steps}  Learning Rate:{scheduler.get_last_lr()[0]:.6f}\n")

    torch.save(model.state_dict(), f"/home/u2024110507/Ref_AVS/saved_model/{args.name}.pth")
    print(f"trained model saved as {args.name}.pth")

    # ---------------在 seen 和 unseen 上测试------------------------------------------
    model.eval()

    valuate(model, val_dataloader_s_refer, args, 'test_s_refer')
    valuate(model, val_dataloader_u_refer, args, 'test_u_refer')

    # ---------------在 Null 上测试------------------------------------------
    model.eval()

    total_metric = 0
    count = 0

    for batch in tqdm(val_dataloader_n_refer, desc=f"Evaluating on test_n_refer"):
        input_dict = dict_to_cuda(batch)
        with torch.no_grad():
            output_dict = model.forward(images=input_dict["images"],
                                        images_clip=input_dict["images_clip"],
                                        audio_features=input_dict["audio_feats"],
                                        image_features=input_dict["image_feats"],
                                        input_ids=input_dict["input_ids"],
                                        labels=input_dict["labels"],
                                        attention_masks=input_dict["attention_masks"],
                                        masks_list=input_dict["masks"],
                                        resize_list=input_dict["resizes"],
                                        orgsize_list=input_dict["orgsizes"],
                                        conversation_list=input_dict["convs"],
                                        refs_num=input_dict["refs_num"],
                                        fids=input_dict["fids"],
                                        vids=input_dict["vids"],
                                        contrast=args.ct_weight,
                                        ref_ids=input_dict["ref_ids"],
                                        inference=True)
        pred_masks = output_dict["pred_masks"]  # list[B]:[num_seg, T, H, W]
        gt_masks = output_dict["gt_masks"]  # list[B]:[num_seg, T, H, W]
        for i in range(len(pred_masks)):
            num_seg = pred_masks[i].shape[0]
            T = pred_masks[i].shape[1]
            null_metric = utility.metric_s_for_null(pred_masks[i])

            total_metric += null_metric * num_seg * T
            count += num_seg * T


    print(f"\n  valuate on test_n_refer, metric: {total_metric/count}")


    with open(f"/home/u2024110507/Ref_AVS/checkpoints/{args.name}.txt", "a") as f:
        f.write(f"\n valuate on  test_n_refer:   metric {total_metric/count} \n")