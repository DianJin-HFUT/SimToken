import pandas as pd
from towhee import pipe, ops
import torch



audio_vggish_pipeline = (  # pipeline building
     pipe.input('path')
     .map('path', 'frame', ops.audio_decode.ffmpeg())
     .map('frame', 'vecs', ops.audio_embedding.vggish())
     .output('vecs')
)

data_dir = '/home/u2024110507/Ref_AVS/data'

metapath = '/home/u2024110507/Ref_AVS/data/metadata.csv'
metadata = pd.read_csv(metapath, header=0)
metadata = metadata[metadata['split'].isin(['train', 'val', 'test_s', 'test_u', 'test_n'])]

vids = metadata['uid'].apply(lambda x: x.rsplit('_', 2)[0]).unique()


for vid in vids:
    audio_path = f'{data_dir}/media/{vid}/audio.wav'
    audio_embed = torch.tensor(audio_vggish_pipeline(audio_path).get()[0])
    torch.save(audio_embed, f'{data_dir}/audio_embed/{vid}.pt')
    print(f'{vid} embedding saved {audio_embed.shape}')

