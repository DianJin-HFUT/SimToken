from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from PIL.ImImagePlugin import split

from .multimodal_encoder.builder import build_vision_tower
from ChatUniVi.constants import *
from .cluster import CTM, TCBlock
from collections import OrderedDict
from .multimodal_projector.builder import build_vision_projector


class MetaModel:
    def __init__(self, config):
        super(MetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

        if hasattr(config, "config"):
            self.use_cluster = config.config["use_cluster"]
            if self.use_cluster:
                self.ctm0 = CTM(sample_ratio=config.config["spatial_cluster_rate0"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=5)
                self.block0 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

                self.ctm1 = CTM(sample_ratio=config.config["spatial_cluster_rate1"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=3)
                self.block1 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

                self.ctm2 = CTM(sample_ratio=config.config["spatial_cluster_rate2"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=3)
                self.block2 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

                self.ctm3 = CTM(sample_ratio=config.config["temporal_cluster_rate"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=5)
                self.block3 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)
        else:
            self.use_cluster = False

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_cluster_modules(self, model_args):
        self.use_cluster = model_args.use_cluster

        if self.use_cluster and not hasattr(self, 'ctm0'):
            self.ctm0 = CTM(sample_ratio=model_args.spatial_cluster_rate0, embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=5)
            self.block0 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

            self.ctm1 = CTM(sample_ratio=model_args.spatial_cluster_rate1, embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=3)
            self.block1 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

            self.ctm2 = CTM(sample_ratio=model_args.spatial_cluster_rate2, embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=3)
            self.block2 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

            self.ctm3 = CTM(sample_ratio=model_args.temporal_cluster_rate, embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=5)
            self.block3 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)


class ChatUniViMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images, select_feature="patch")
        return image_features

    def positional_encoding(self, x, num_features=1024, max_len=64):
        p = torch.zeros((1, max_len, num_features))
        _x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000,
                                                                            torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)

        p[:, :, 0::2] = torch.sin(_x)
        p[:, :, 1::2] = torch.cos(_x)
        x = x + p[:, :x.shape[1], :].to(x.device).to(x.dtype)
        return x

    def project(self, image_features, input_type="image"):
        if self.get_model().use_cluster:
            if input_type == "image":
                cluster_image_features = []
                token_dict = {'x': image_features,
                              'token_num': image_features.size(1),
                              'idx_token': torch.arange(image_features.size(1))[None, :].repeat(
                                  image_features.size(0), 1),
                              'agg_weight': image_features.new_ones(image_features.size(0), image_features.size(1),
                                                                    1),
                              'mask': None}

                token_dict = self.get_model().block0(self.get_model().ctm0(token_dict))
                cluster_image_features.append(token_dict["x"])

                token_dict = self.get_model().block1(self.get_model().ctm1(token_dict))
                cluster_image_features.append(token_dict["x"])

                token_dict = self.get_model().block2(self.get_model().ctm2(token_dict))
                cluster_image_features.append(token_dict["x"])

                image_features = torch.cat(cluster_image_features, dim=1)
                image_features = image_features.to(self.get_model().mm_projector.weight.dtype)
            else:
                cls_features = torch.mean(image_features, dim=1, keepdim=False).unsqueeze(0).clone()
                token_dict = {'x': cls_features,
                              'token_num': cls_features.size(1),
                              'idx_token': torch.arange(cls_features.size(1))[None, :].repeat(
                                  cls_features.size(0), 1),
                              'agg_weight': cls_features.new_ones(cls_features.size(0), cls_features.size(1),
                                                                  1),
                              'mask': None}

                down_dict, token_dict = self.get_model().ctm3(token_dict)
                events = OrderedDict()

                max_len = 0
                for id, i in enumerate(down_dict["idx_token"][0].tolist()):
                    if i not in events:
                        events[i] = [id]
                    else:
                        events[i].append(id)
                    max_len = len(events[i]) if max_len < len(events[i]) else max_len

                cluster_image_features = []
                token_dict = {'x': image_features,
                              'token_num': image_features.size(1),
                              'idx_token': torch.arange(image_features.size(1))[None, :].repeat(
                                  image_features.size(0), 1),
                              'agg_weight': image_features.new_ones(image_features.size(0), image_features.size(1),
                                                                    1),
                              'mask': None}

                token_dict0 = self.get_model().block0(self.get_model().ctm0(token_dict))
                token_dict1 = self.get_model().block1(self.get_model().ctm1(token_dict0))
                token_dict2 = self.get_model().block2(self.get_model().ctm2(token_dict1))

                for id, key in enumerate(events):
                    cur_image_features0 = torch.cat([token_dict0["x"][i] for i in events[key]], dim=0).unsqueeze(0)
                    token_dict = {'x': cur_image_features0,
                                  'token_num': cur_image_features0.size(1),
                                  'idx_token': torch.arange(cur_image_features0.size(1))[None, :].repeat(
                                      cur_image_features0.size(0), 1),
                                  'agg_weight': cur_image_features0.new_ones(cur_image_features0.size(0),
                                                                             cur_image_features0.size(1),
                                                                      1),
                                  'mask': None}

                    cur_token_dict0 = self.get_model().block0(self.get_model().ctm0(token_dict))
                    cluster_image_features.append(cur_token_dict0["x"])

                    cur_image_features1 = torch.cat([token_dict1["x"][i] for i in events[key]], dim=0).unsqueeze(0)
                    token_dict = {'x': cur_image_features1,
                                  'token_num': cur_image_features1.size(1),
                                  'idx_token': torch.arange(cur_image_features1.size(1))[None, :].repeat(
                                      cur_image_features1.size(0), 1),
                                  'agg_weight': cur_image_features1.new_ones(cur_image_features1.size(0),
                                                                             cur_image_features1.size(1),
                                                                             1),
                                  'mask': None}

                    cur_token_dict1 = self.get_model().block1(self.get_model().ctm1(token_dict))
                    cluster_image_features.append(cur_token_dict1["x"])

                    cur_image_features2 = torch.cat([token_dict2["x"][i] for i in events[key]], dim=0).unsqueeze(0)
                    token_dict = {'x': cur_image_features2,
                                  'token_num': cur_image_features2.size(1),
                                  'idx_token': torch.arange(cur_image_features2.size(1))[None, :].repeat(
                                      cur_image_features2.size(0), 1),
                                  'agg_weight': cur_image_features2.new_ones(cur_image_features2.size(0),
                                                                             cur_image_features2.size(1),
                                                                             1),
                                  'mask': None}

                    cur_token_dict2 = self.get_model().block2(self.get_model().ctm2(token_dict))
                    cluster_image_features.append(cur_token_dict2["x"])

                image_features = torch.cat(cluster_image_features, dim=1)
                image_features = image_features.to(self.get_model().mm_projector.weight.dtype)

        else:
            if input_type == "video":
                image_features, cls_features = torch.mean(image_features, dim=0, keepdim=False).unsqueeze(
                    0), torch.mean(image_features, dim=1, keepdim=False).unsqueeze(0)
                image_features = torch.cat([image_features, cls_features], dim=1)

        image_features = self.get_model().mm_projector(image_features)
        return image_features # 不同的type形状相同

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, audio_features=None, target_frame=0, ref_ids=None
    ):
        IMAGE_TOKEN_INDEX = -200
        AUDIO_TOKEN_INDEX = -300
        # print("\n调用prepare_inputs_labels_for_multimodal")
        vision_tower = self.get_vision_tower()
        # print("获取vision_tower")
        num_frames = images[0].shape[0]  # T


        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels


        if ref_ids is not None:
            ref_embeds = []
            for ref_id in ref_ids:
                ref_embed = self.get_model().embed_tokens(ref_id) #[L, 4096]
                ref_embeds.append(ref_embed)
            # list[B]: [len_ref, 4096]





        if type(images) is list or images.ndim == 5:
            # print("先concat列表中的图像")
            concat_images = torch.cat([image for image in images], dim=0)  # [BT, 3, H, W]
            org_image_features = self.encode_images(concat_images)  # [BT, 256, 1024]

            # if audio_features is not None and hasattr(self, "audio_adapter"):
            if True:
                # image_features = self.audio_adapter(org_image_features, audio_features, ref_embeds_T)
                # image_features = self.token_compressor(org_image_features, ref_embeds)
                # print("image_features after compress:", image_features.shape)
                image_features = org_image_features

            else:
                image_features = org_image_features
            # split_sizes = [image.shape[0] for image in images]
            split_sizes = 1
            image_features = torch.split(image_features, split_sizes, dim=0)  # list[BT]: [1, 256,1024]
            image_features = [x.flatten(0, 1) for x in image_features] # list[BT]: [256,1024]

            org_image_features = torch.split(org_image_features, split_sizes, dim=0)
            org_image_features = [x.flatten(0, 1) for x in org_image_features]

        else:
            # print("直接获取image_feature")
            image_features = self.encode_images(images)
            org_image_features = image_features



        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # cur_image_idx += 1

            # 判断当前input_id中有没有图像token
            # print("cur_input_ids shape:", cur_input_ids.shape)
            # print("cur_input_ids:", cur_input_ids)
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # print("input_ids中没有 IMAGE token")
                # multimodal LLM, but the current sample is not multimodal
                # 直接把input_ids进行text embed
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (
                            0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where((cur_input_ids == IMAGE_TOKEN_INDEX)|(cur_input_ids == AUDIO_TOKEN_INDEX))[0]
            audio_token_indices = torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0]
            # print("audio indices:", audio_token_indices)
            # print("image and audio indices:", image_token_indices)

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape


            # 有多个image token---------------------------------------------
            if len(image_token_indices) > 1:
                # print("有多个image token")
                # return 0

                temp = []

                cur, pre = image_token_indices[0], image_token_indices[0]
                # 这里是把连续的<image>的位置放到一个list中存储 分割开的<image>
                for i in image_token_indices:
                    cur = i
                    # 如果下一个<image>就在上一个<image>之后
                    if cur - pre == 1:
                        temp[-1] = temp[-1] + [cur]
                    else:
                        temp.append([cur])
                    pre = cur


                pre_image_token_end = 0
                cur_frames = 0
                for i in temp:
                    # 第一个以及最后一个<image>的位置
                    image_token_start = i[0]
                    image_token_end = i[-1]
                    cur_image_features = []

                    if len(i) >= 2:  # 处理T个image组成的视频特征
                        for frame_idx in range(num_frames):
                            cur_image_features.append(org_image_features[batch_idx*num_frames+frame_idx])
                            # print(batch_idx*num_frames+frame_idx)
                    elif i[0] not in audio_token_indices:
                        cur_image_features.append(org_image_features[batch_idx * num_frames + target_frame])
                        # print(batch_idx * num_frames + target_frame)
                    else:
                        cur_image_features.append(audio_features[batch_idx])
                        # print(f"audio{batch_idx}")
                    # ------------------------------------------------------------------
                    # # i是每组<image>的indices 根据其数量从image_features中拿特征
                    # for _ in i:
                    #     # 表示处理的是<image>
                    #     if _ not in audio_token_indices:
                    #         # 单个image
                    #         if cur_frames == num_frames:
                    #             # cur_image_features.append(org_image_features[cur_image_idx-num_frames+target_frame])
                    #             cur_image_features.append(org_image_features[batch_idx*num_frames+target_frame])
                    #             # print(cur_image_idx-num_frames+target_frame)
                    #         # 多个image
                    #         else:
                    #             cur_image_features.append(image_features[cur_image_idx])
                    #             # print(cur_image_idx)
                    #             cur_image_idx += 1
                    #         cur_frames += 1
                    #     # 处理<audio>
                    #     else:
                    #         # cur_image_features.append(self.audio_feature_layer(audio_features[batch_idx]))
                    #         cur_image_features.append(audio_features[batch_idx])
                    #         # print("audio:", batch_idx)
                    # # cur_image_features list[len(i)] : [256,1024]



                    # 如果当前分组是多个image 代表video
                    if len(i) >= 2:
                        if not self.compress:

                            # 对拿到的多个image_features进行压缩 并投影
                            cur_image_features = torch.stack(cur_image_features, dim=0)  # [len(i), 256, 1024]
                            cur_image_features = self.project(cur_image_features, input_type="video")
                            t, l, n = cur_image_features.size()
                            cur_image_features = cur_image_features.contiguous().view(t * l, n) #[112, 4096]
                            # print(f"no compression, cur_image_features{cur_image_features.shape}")

                        else:

                            compressed_frames = []
                            for cur_image_feature in cur_image_features:
                                cur_image_feature = self.project(cur_image_feature.unsqueeze(0), input_type="image")  # [1, 256, 1024]
                                t, l, n = cur_image_feature.size()
                                cur_image_feature = cur_image_feature.contiguous().view(t * l, n)  # [112, 4096]

                                compressed_frames.append(cur_image_feature.mean(dim=0).unsqueeze(0))  # [1, 4096]
                            compressed_frames = torch.cat(compressed_frames, dim=0)  # [T, 4096]

                            cur_image_features = torch.stack(cur_image_features, dim=0)  # [len(i), 256, 1024]
                            cur_image_features = self.project(cur_image_features, input_type="video")
                            t, l, n = cur_image_features.size()
                            cur_image_features = cur_image_features.contiguous().view(t * l, n)  # [112, 4096]

                            # cur_image_features = torch.cat([cur_image_features, compressed_frames], dim=0)  # [122, 4096]
                            cur_image_features = torch.cat([compressed_frames, cur_image_features], dim=0)  # [122, 4096]

                    # 对于单个的特殊 token 如果是<image>
                    elif i[0] not in audio_token_indices:
                        cur_image_features = torch.stack(cur_image_features, dim=0)
                        cur_image_features = self.project(cur_image_features, input_type="image")
                        t, l, n = cur_image_features.size()
                        cur_image_features = cur_image_features.contiguous().view(t * l, n)  # [112, 4093]
                    else:
                        cur_image_features = cur_image_features[0]  #[10, 4096]


                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                        # 把im_start前的文字进行embeds
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[pre_image_token_end:image_token_start - 1]).detach())
                        # 把im_start进行embeds
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                        # 图像特征
                        cur_new_input_embeds.append(cur_image_features)
                        # im_end
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_end + 1:image_token_end + 2]))
                        if labels is not None:
                            cur_new_labels.append(cur_labels[pre_image_token_end:image_token_start])
                            # cur_new_labels填充
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                            cur_new_labels.append(cur_labels[image_token_end:image_token_end + 1])

                            # cur_labels设置为剩余的cur_labels
                            # cur_labels = cur_labels[image_token_end + 2:]
                    else:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[pre_image_token_end:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        if labels is not None:
                            cur_new_labels.append(cur_labels[pre_image_token_end:image_token_start])
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                            # cur_labels = cur_labels[image_token_end + 1:]

                    pre_image_token_end = image_token_end + 1


                # cur_input_ids设置为剩余的cur_input_ids
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_input_ids = cur_input_ids[image_token_end + 2:]
                    cur_labels = cur_labels[image_token_end + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_end + 1:]
                    cur_labels = cur_labels[image_token_end + 1:]

            # 结合上面大于1 此处就是只有一个image token
            elif image_token_indices.numel() > 0:
                # print("只有一个image token")

                cur_image_features = []
                image_token_start = image_token_indices[0]
                image_token_end = image_token_indices[-1]
                # print("image_token_start:", image_token_start, " image_token_end:", image_token_end)

                # 根据image token数量 把image feature加入到cur_image_features
                for _ in image_token_indices:
                    cur_image_features.append(image_features[cur_image_idx])
                    cur_image_idx += 1
                # print("cur_image_features length:", len(cur_image_features))

                # 对image features进行维度上拼接
                cur_image_features = torch.stack(cur_image_features, dim=0)
                # print("cur_image_features_stacked shape:", cur_image_features.shape)
                cur_image_features = self.project(cur_image_features, input_type="image")
                # print("cur_image_features_projected shape:", cur_image_features.shape)

                # 获取 图像特征的维度 nums, len, dim
                t, l, n = cur_image_features.size()
                cur_image_features = cur_image_features.contiguous().view(t * l, n)
                # print("cur_image_features_viewed shape:", cur_image_features.shape)



                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    print("no tune_mm_mlp_adapter and no mm_use_im_start_end")
                    # 把imagetoken之前的text进行embedding 这两行
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    # 这里加入的 image——strat——token
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    print("cur_new_input_embeds length:", len(cur_new_input_embeds))
                    print("cur_new_input_embeds shape:", cur_new_input_embeds[0].shape)
                    print("cur_new_input_embeds shape:", cur_new_input_embeds[1].shape)

                    # 在图像token位置上加入image feature
                    cur_new_input_embeds.append(cur_image_features)
                    print("cur_new_input_embeds length:", len(cur_new_input_embeds))
                    # print("cur_new_input_embeds shape:", cur_new_input_embeds[2].shape)

                    # 把图像token之后的img-end-token加入
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_end+1:image_token_end+2]))
                    print("cur_new_input_embeds length:", len(cur_new_input_embeds))

                    if labels is not None:
                        # 把image token前面的label加入
                        cur_new_labels.append(cur_labels[:image_token_start])
                        # 根据图像特征形状加入 多个ignore index
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        # 把img-end-token加入
                        cur_new_labels.append(cur_labels[image_token_end:image_token_end+1])
                        # 把剩下的text label加入
                        cur_labels = cur_labels[image_token_end+2:]

                else:
                    # print("tune_mm_mlp_adapter / mm_use_im_start_end")
                    # 对图像token之前的text token 进行embedding
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    # print("cur_new_input_embeds length:", len(cur_new_input_embeds))

                    if labels is not None:
                        # 把图像前面的labels进行复制
                        cur_new_labels.append(cur_labels[:image_token_start])
                        # 根据图像特征形状 加入shape[0]个 IGNORE_INDEX
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        # 加入剩下的labels
                        # print("cur_new_labels length:", len(cur_new_labels))
                        # print("cur_new_labels:", cur_new_labels)
                        # print(cur_new_labels[0].shape, '   ',cur_new_labels[1].shape)

                        # 将cur_labels保留为剩下的未处理过的lables
                        cur_labels = cur_labels[image_token_end+1:]
                        # print("labels after image:", cur_labels)
                        # print(len(cur_labels))


                # 将 cur_input_ids替换为剩下的 没有处理的 (img之后的) ids
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_end+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_end+1:]
                # print("input_ids after image :", cur_input_ids)

            # 如果图像token之后还有text token
            if cur_input_ids.numel() > 0:
                # print("image token 之后还有 text token")
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    # 把剩下的input_id进行embedding

                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))

                    # print("cur_new_input_embeds length:", len(cur_new_input_embeds))
                    # print("cur_new_input_embeds shape:", cur_new_input_embeds[0].shape, cur_new_input_embeds[1].shape, cur_new_input_embeds[2].shape)

                if labels is not None:
                    # 把剩下的labels加入
                    cur_new_labels.append(cur_labels)


            cur_new_input_embeds = [x.to(device='cuda') for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)

            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)

                new_labels.append(cur_new_labels)

        # 如果一个batch内部embedd inputs长度不一致
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            print("batch 内部长度不一致")
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape

        # 内部长度一致
        else:
            # 将一个batch的数据 拼接成 [B, token_len, dim]
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False