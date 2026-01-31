import os.path

from models.segment_anything import build_sam_vit_h
from models.segment_anything.utils.transforms import ResizeLongestSide
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from configs import args
from save_audio_feats import data_dir


def preprocess(x: torch.Tensor, device='cuda') -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # 确保输入张量在正确的设备上
    x = x.to(device)

    # Normalize colors
    pixel_mean = torch.Tensor([113.263, 99.370, 92.492]).view(-1, 1, 1).to(device)
    pixel_std = torch.Tensor([64.274, 61.068, 58.626]).view(-1, 1, 1).to(device)
    img_size = 1024

    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_dir = args.data_dir
metapath = os.path.join(data_dir, 'metadata.csv')
metadata = pd.read_csv(metapath, header=0)
metadata = metadata[metadata['split'].isin(['train', 'val', 'test_s', 'test_u', 'test_n'])]
# metadata = metadata[metadata['split'].isin(['test_s'])]

vids = metadata['uid'].apply(lambda x: x.rsplit('_', 2)[0]).unique()

sam_model = build_sam_vit_h(args.vision_pretrained)
sam_model.to(device)

for param in sam_model.parameters():
    param.requires_grad = False

save_dir = os.path.join(data_dir, 'image_embed')
os.makedirs(save_dir, exist_ok=True)


torch.cuda.empty_cache()

for vid in vids:
    image_embeds = []

    for _idx in range(10):
        path_frame = f'{data_dir}/media/{vid}/frames/{_idx}.jpg'
        frame = cv2.imread(path_frame)
        if frame is None:
            print(f"Warning: Could not read image {path_frame}")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = ResizeLongestSide(1024).apply_image(frame)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous()  # [3, H, W]

        frame_processed = preprocess(frame_tensor, device)  # [3, 1024, 1024]

        single_image = frame_processed.unsqueeze(0)  # [1, 3, 1024, 1024]

        with torch.no_grad():
            image_embed = sam_model.image_encoder(single_image)  # [1, 256, 64, 64]
            image_embed = image_embed.squeeze(0).cpu()

        image_embeds.append(image_embed)


        torch.cuda.empty_cache()

    if not image_embeds:
        print(f"Error: No images loaded for video {vid}")
        continue


    image_embeds_stacked = torch.stack(image_embeds, dim=0)  # [T, 256, 64, 64]


    torch.save(image_embeds_stacked, f'{save_dir}/{vid}.pt')

    print(f"Processed video {vid}, features shape: {image_embeds_stacked.shape}")

print("Processing completed!")