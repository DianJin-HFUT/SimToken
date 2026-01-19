from models.segment_anything import build_sam_vit_h
from models.segment_anything.utils.transforms import ResizeLongestSide
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

def preprocess(x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean = torch.Tensor([113.263, 99.370, 92.492]).view(-1, 1, 1)
    pixel_std = torch.Tensor([64.274, 61.068, 58.626]).view(-1, 1, 1)
    img_size = 1024

    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


data_dir = '/home/u2024110507/Ref_AVS/data'
metapath = '/home/u2024110507/Ref_AVS/data/metadata.csv'
metadata = pd.read_csv(metapath, header=0)
metadata = metadata[metadata['split'].isin(['train', 'val', 'test_s', 'test_u', 'test_n'])]
vids = metadata['uid'].apply(lambda x: x.rsplit('_', 2)[0]).unique()

sam_model = build_sam_vit_h("/home/u2024110507/Ref_AVS/models/segment_anything/sam_vit_h_4b8939.pth")
for param in sam_model.parameters():
    param.requires_grad = False

for vid in vids:
    image = []
    for _idx in range(10):
        path_frame = f'{data_dir}/media/{vid}/frames/{_idx}.jpg'
        frame = cv2.imread(path_frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = ResizeLongestSide(1024).apply_image(frame)
        frame = preprocess(torch.from_numpy(frame).permute(2, 0, 1).contiguous())  # [3, 1024, 1024]
        image.append(frame)
    image = torch.stack(image, dim=0)  # [10, 3, 1024, 1024]
    with torch.no_grad():
        image_embed = sam_model.image_encoder(image) # [T, 256, 64, 64]]
    # print(images_embs.shape)
    # break
    torch.save(image_embed, f'{data_dir}/image_embed/{vid}.pt')
    # break


