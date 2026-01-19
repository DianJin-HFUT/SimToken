from turtledemo.penrose import start
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

# from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN

from ChatUniVi.model.language_model.llama import ChatUniViLlamaForCausalLM, ChatUniViLlamaModel

from models.segment_anything import build_sam_vit_h
from ChatUniVi.constants import IMAGE_TOKEN_INDEX

import cv2
import time
import random
import math
from collections import defaultdict



def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        scale: float = 1000,
        eps: float = 1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss



def compute_alignment_loss(q: torch.Tensor, pos_feats: list, neg_feats: list, temperature=0.07):
    """
    q: [B, D] embedding of the output SEG token
    pos_feats: List[B][List[Tensor[D]]]   semantic embeddings of positive sets
    """
    B, D = q.shape
    device = q.device
    total_loss = 0.0
    count = 0

    for i in range(B):
        pos = pos_feats[i]
        neg = neg_feats[i]

        if len(pos) == 0:
            continue

        # === Normalize ===
        anchor = F.normalize(q[i].unsqueeze(0), dim=1)  # [1, D]
        pos_tensors = torch.stack(pos).to(device)  # [P, D]
        pos_tensors = F.normalize(pos_tensors, dim=1)    # [P, D]

        # === Alignment ===
        sim_pos = torch.matmul(anchor, pos_tensors.T) / temperature  # [1, P]
        log_probs = F.log_softmax(sim_pos, dim=1)
        loss = -log_probs.mean()
        total_loss += loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / count



# 模型的基础类，负责初始化视觉模块和投影层
class Simtoken_MetaModel:
    def __init__(
            self,
            config,
            **kwargs,
    ):
        super(Simtoken_MetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class Simtoken_Model(Simtoken_MetaModel, ChatUniViLlamaModel):
    def __init__(
            self,
            config,
            **kwargs,
    ):
        super(Simtoken_Model, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class SemanticMemoryBank:
    def __init__(self, max_per_object=5):
        self.bank = defaultdict(lambda: defaultdict(list))  # bank[vid][fid] = [feat1, feat2, ...]
        self.max_per_object = max_per_object

    def add(self, vid: str, fid: int, feat: torch.Tensor):
        """添加一条特征，并限制每个 (vid, fid) 对应的特征数量不超过 max_per_object"""
        feat = feat.detach().cpu()
        self.bank[vid][fid].append(feat)
        if len(self.bank[vid][fid]) > self.max_per_object:
            self.bank[vid][fid] = self.bank[vid][fid][-self.max_per_object:]  # 保留最新的 K 个

    def add_batch(self, vids: list, fids: list, feats: torch.Tensor):
        """添加一批数据：vids: list[B], fids: list[B], feats: Tensor[B, D]"""
        for vid, fid, feat in zip(vids, fids, feats):
            self.add(vid, int(fid), feat)

    def get_positive_features(self, vids: list, fids: list):
        """返回 List[B]，每个元素是 List[Tensor]，表示该样本的正样本特征（同vid+fid）"""
        results = []
        for vid, fid in zip(vids, fids):
            pos = self.bank[vid][int(fid)].copy()  # List[Tensor]
            results.append(pos)
        return results

    def get_negative_features_same_vid(self, vids: list, fids: list):
        """返回 List[B]，每个元素是 List[Tensor]，表示该样本在同一视频中其他 fid 的负样本"""
        results = []
        for vid, fid in zip(vids, fids):
            neg = []
            for other_fid, feats in self.bank[vid].items():
                if other_fid != int(fid):
                    neg.extend(feats)
            results.append(neg)
        return results


# class VISAForCausalLM(ChatUniViLlamaForCausalLM):
class Simtoken_ForCausalLM(ChatUniViLlamaForCausalLM):
    def __init__(
            self,
            config,
            **kwargs,
    ):
        # 判断 config 是否包含 train_mask_decoder
        if not hasattr(config, "train_mask_decoder"):
            #
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            # 获取参数中的vision_tower 没获取到默认clip
            config.mm_vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14")
            # 从 kwargs 字典中取出 weight 的值，。如果 kwargs 里没有 eight，则返回 None
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower

        self.seg_token_idx = kwargs.pop("seg_token_idx")


        super().__init__(config)

        self.model = Simtoken_Model(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.audio_feature_layer = nn.Linear(in_features=128, out_features=4096)

        self.memory = SemanticMemoryBank()

        self.compress = kwargs.pop("compress", True)

        self.start = kwargs.pop("start")



    # 提取图像embedding 注意这里不需要梯度 即是冻结的
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            # 这里的self.model是上面的VisaModel
            image_embeddings = self.model.visual_model.image_encoder(pixel_values)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
            self,
            images: torch.FloatTensor,
            images_clip: torch.FloatTensor,
            audio_features: torch.FloatTensor,
            image_features: torch.FloatTensor,
            input_ids: torch.LongTensor,
            labels: torch.LongTensor,
            attention_masks: torch.LongTensor,
            masks_list: List[torch.FloatTensor],
            resize_list: List[tuple],
            orgsize_list: List[tuple],
            conversation_list: List[str],
            # num_frame_list: List[int], # 固定为10
            # num_conv_list: List[int],  # 固定为1
            ref_ids: List[torch.LongTensor],
            refs_num: List[int],
            vids,
            fids,
            epoch: int =0,
            inference: bool = False,
            num_frames: int = 10,
            contrast: float = 0.0,

            **kwargs,
    ):
        batch_size = len(images)
        image_embeddings = torch.cat(image_features, dim=0)
        # image_embeddings = self.get_visual_embs(torch.cat(images, dim=0)) # [BT, 256, 64, 64]

        audio_embeddings = self.audio_feature_layer(torch.stack(audio_features, dim=0))  # [B, 10, 4096]
        # audio_embeddings = torch.cat(audio_features, dim=0) # [B*10, 128]
        # audio_embeddings = audio_features  # [B, 10, 128]

        # train
        if not inference:
            target_frame = random.randint(0, 9)
            target_frame = 5

        else:
            target_frame = 5
        # print("target_frame", target_frame)

        input_ids, attention_masks, past_key_values, inputs_embeds, labels = super().prepare_inputs_labels_for_multimodal(
            input_ids, attention_masks, past_key_values=None, labels=labels, images=images_clip, audio_features=audio_embeddings, target_frame=target_frame, ref_ids=ref_ids
        )


        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_masks,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states
        # print("last layer of output_hidden_states:", output_hidden_states[-1].shape)  # [B, len, 4096]

        # <seg>要提前一位
        seg_token_mask = output.labels[..., 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [seg_token_mask, torch.zeros((seg_token_mask.shape[0], 1), device=output.labels.device).bool(), ],
            dim=1, )  # [batch_size, seq_len]


        seg_embeddings = self.model.text_hidden_fcs[0](output_hidden_states[-1][seg_token_mask])  # [seg_num,256]

        # print("seg_embeddings in this batch:", seg_embeddings.shape)
        # print("vids:", vids)
        # print("fids:", fids)
        fis_flat = [fid[0] for fid in fids]
        # print("fids:", fis_flat )
        if not inference:

            pos_feats = self.memory.get_positive_features(vids, fis_flat )
            neg_feats = self.memory.get_negative_features_same_vid(vids, fis_flat )

            for i in range(len(neg_feats)):
                for j in range(len(seg_embeddings)):
                    if j != i:
                        neg_feats[i].append(seg_embeddings[j].detach().cpu())

            ct_loss = compute_alignment_loss(seg_embeddings, pos_feats, neg_feats)

            # print("ct loss:", ct_loss)
            self.memory.add_batch(vids, fis_flat, seg_embeddings)


        pred_embeddings = []
        #--------------------------------------------------------------------------------------------
        pred_idx = 0
        for ref_num in refs_num:
            pred_embeddings.append(seg_embeddings[pred_idx:pred_idx + ref_num])
            pred_idx += ref_num
        # list[B]:[num_seg, 256]


        pred_masks = []

        # 遍历batch
        for i in range(batch_size):

            (
                sparse_embeddings,  # [num_seg, 1, 256]
                dense_embeddings,   # [num_seg, 256, 64, 64]
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),  # [1, 1 ,256]
            )
            # 确保数据类型一致
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            dense_embeddings = dense_embeddings.to(pred_embeddings[i].dtype)
            # print("sparse_embeddings:", sparse_embeddings.shape)
            # print("dense_embeddings:", dense_embeddings.shape)


            pred_masks_sample = []
            # 遍历当前样本的所有seg:
            for prompt_idx in range(len(sparse_embeddings)):
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i * num_frames: (i + 1) * num_frames],  # [T, 256, 64, 64]
                    # 位置编码 形状与 image_embeddings 一致。
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings[prompt_idx : prompt_idx+1],
                    dense_prompt_embeddings=dense_embeddings[prompt_idx : prompt_idx+1],
                    multimask_output=False,
                )  # low_res_masks [T, 1, 256, 256]

                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=orgsize_list[i]
                )  # [T, 1, H, W]
                pred_masks_sample.append(pred_mask.squeeze(1))  # list[num_seg]:[[T, H, W]]

            pred_masks_sample = torch.stack(pred_masks_sample, dim=0)  # [num_seg, T, H, W]
            pred_masks.append(pred_masks_sample)  # list[B]:[num_seg, T, H, W]




        gt_masks = masks_list # list[B]:[num_seg, T, H, W]

        if inference:
            return {
                "pred_masks": pred_masks,  # list[B]:[num_seg, T, H, W]
                "gt_masks": gt_masks,  # list[B]:[num_seg, T, H, W]
            }

        model_output = output
        output = model_output.logits

        # 计算各种loss
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        # 计算预测掩码和gt之间的loss
        for batch_idx in range(batch_size):


            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            a, b, c, d = gt_mask.shape
            gt_mask = gt_mask.view(a*b, c, d)  # [num_ref*T, H, W]
            pred_mask = pred_mask.view(a*b, c, d)  # [num_ref*T, H, W]

            # print("gt_mask:", gt_mask.shape)

            # 计算mask_bceloss
            mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )
            mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        # ct_losses = []
        # for sample_idx, pred_embedding in enumerate(pred_embeddings):
        #     ct_loss_sample = compute_supcon_loss(pred_embedding, fids[sample_idx])
        #     if ct_loss_sample is not None:
        #         ct_losses.append(ct_loss_sample)
        # if len(ct_losses) > 0:
        #     ct_loss = torch.stack(ct_losses).mean()
        # else:
        #     ct_loss = torch.tensor(0.0, requires_grad=True)

        ct_weight = contrast
        # print("ct_weight", ct_weight)

        if epoch >= self.start:
            loss = ce_loss + mask_loss + ct_weight * ct_loss
        else:
            loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "ct_loss": ct_loss,
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
        }


    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("This method is not implemented.")

