# SimToken: A Simple Baseline for Referring Audio-Visual Segmentation
[![TGS](https://img.shields.io/badge/Paper-SimToken-red?logo=arXiv)](https://arxiv.org/abs/2509.17537)

---
## 📰 News

[//]: # (🔥**2026.1.18**: Code are released now！)

🔥**2026.1.18**: Our paper got accepted to **ICASSP 2026**! Thanks to all co-authors and the anonymous reviewers🎉🎉

---
## ⚙️ Setup

### Datasets

Download the official Ref-AVSBench dataset from [here](https://github.com/GeWu-Lab/Ref-AVS) and organize the dataset as follows:
```
./REFAVS/data 
    - /media 
    - /gt_mask 
    - /metadata.csv 
```

### Pretrained Backbones
Download the sam_vit_h_4b8939.pth and put it in ```./models/segment_anything```

### Checkpoints

Download our pretrained  **[Simtoken](https://drive.google.com/drive/folders/1S-AO0Jqb6zVN1ABHH9VBpZQYJ8PQ22ES?usp=drive_link)**.

### Requirements
Use  ```simtoken.yml``` to create your environment.

---
## 📌 Getting Started

### Preparation
- We recommend running the following code to pre-extract audio features and visual features compatible with SAM:
```
python save_audio_feats.py
python save_sam_feats.py
```
- And organize the obtained features as:
```
./REFAVS/data 
    - /... 
    - /... 
    - /...
    - /audio_embed
    - /image_embed
```

### Train 
```
python -W ignore train.py --name 'xxx' \
    --vision_pretrained 'path/to/segment_anything/sam_vit_h_4b8939.pth' \
    --vision_tower 'openai/clip-vit-large-patch14' \
    --mllm 'Chat-UniVi/Chat-UniVi' \
    --data_dir 'path/to/data'

```
### Test
```
python -W ignore load_model.py  --saved_model 'path/to/checkpoint.pth' \
    --vision_pretrained 'path/to/segment_anything/sam_vit_h_4b8939.pth' \
    --vision_tower 'openai/clip-vit-large-patch14' \
    --mllm 'Chat-UniVi/Chat-UniVi' \
    --data_dir 'path/to/data' \

```





