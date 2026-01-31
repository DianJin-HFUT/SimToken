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
Download our pretrained  **[Simtoken](https://drive.google.com/file/d/1pargYfFy93rymCANuWV0nt6Lx3Ri406l/view?usp=sharing)**.

### Core Requirements
This project depends on a small set of core packages. The configuration below has been tested and is recommended for stable execution.
- `numpy`, `pandas`, `matplotlib`, `opencv`
- `einops`, `timm`
- `sentencepiece`
- `transformers`, `peft`

Newer versions of `transformers` and `peft` may introduce API changes or naming/registration conflicts that can trigger runtime errors in this project (e.g., custom model/config registration).  
To avoid such compatibility issues, we recommend **not using overly recent versions** and pin the two packages to the versions used during our development:

- `transformers==4.30.2`
- `peft==0.2.0`

We also provide a complete requirements.txt for reference and easier reproduction:
```
pip install -r requirements.txt
```


---
## 📌 Getting Started

### Preparation
We recommend running the following code to pre-extract audio features and visual features compatible with SAM:
```
python save_audio_feats.py --data_dir 'path/to/data'
python save_sam_feats.py  --data_dir 'path/to/data'
```



### Train 
To train our model on Ref-AVS Bench:
```
python -W ignore train.py --name 'xxx' \
    --vision_pretrained 'path/to/segment_anything/sam_vit_h_4b8939.pth' \
    --vision_tower 'openai/clip-vit-large-patch14' \
    --mllm 'Chat-UniVi/Chat-UniVi-7B-v1.5' \
    --data_dir 'path/to/data'\
    --log_root 'path/to/log_root'\
    --checkpoint_root 'path/to/checkpoints_root'

```
### Test
To test our pretrained simtoken:
```
python -W ignore load_model.py  --saved_model 'path/to/checkpoint.pth' \
    --vision_pretrained 'path/to/segment_anything/sam_vit_h_4b8939.pth' \
    --vision_tower 'openai/clip-vit-large-patch14' \
    --mllm 'Chat-UniVi/Chat-UniVi-7B-v1.5' \
    --data_dir 'path/to/data' \
    --visualization_root 'path/to/visualization_root'

```





