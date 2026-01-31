import os.path

import pandas as pd
from towhee import pipe, ops
import torch
from configs import args


audio_vggish_pipeline = (  # pipeline building
     pipe.input('path')
     .map('path', 'frame', ops.audio_decode.ffmpeg())
     .map('frame', 'vecs', ops.audio_embedding.vggish())
     .output('vecs')
)

data_dir = args.data_dir

metapath = os.path.join(data_dir, 'metadata.csv')
metadata = pd.read_csv(metapath, header=0)
metadata = metadata[metadata['split'].isin(['train', 'val', 'test_s', 'test_u', 'test_n'])]
# metadata = metadata[metadata['split'].isin(['test_s'])]

vids = metadata['uid'].apply(lambda x: x.rsplit('_', 2)[0]).unique()

save_dir = os.path.join(data_dir, 'audio_embed')
os.makedirs(save_dir, exist_ok=True)

for vid in vids:
    audio_path = f'{data_dir}/media/{vid}/audio.wav'
    audio_embed = torch.tensor(audio_vggish_pipeline(audio_path).get()[0])
    torch.save(audio_embed, f'{save_dir}/{vid}.pt')
    print(f'{vid} embedding saved {audio_embed.shape}')

