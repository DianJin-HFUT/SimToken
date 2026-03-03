import os.path

import pandas as pd
from towhee import pipe, ops
import torch
from configs import args
import torchaudio
import tempfile


def preprocess_audio_to_mono(input_path, target_sr=16000, keep_original_format=True):

    waveform, sample_rate = torchaudio.load(input_path)
    original_encoding = torchaudio.info(input_path).encoding


    if waveform.shape[0] > 1:
        waveform = waveform[:1, :]

    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)

    if keep_original_format and original_encoding == "PCM_S":
        waveform = (waveform * 32767).to(torch.short)  # float -> int16
        torchaudio.save(temp_path, waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)
    else:
        torchaudio.save(temp_path, waveform, sample_rate)

    return temp_path


audio_vggish_pipeline = (  # pipeline building
     pipe.input('path')
     .map('path', 'frame', ops.audio_decode.ffmpeg())
     .map('frame', 'vecs', ops.audio_embedding.vggish())
     .output('vecs')
)



data_dir = args.data_dir


# test_id = 'zxis5LLvULw_12000_22000'
# test_path = f'{data_dir}/media/{test_id}/audio.wav'
# temp_path = preprocess_audio_to_mono(test_path)
# print(f"original audio info: {torchaudio.info(test_path)}")
# print(f"mono audio info: :{torchaudio.info(temp_path)}")
# test_embed = torch.tensor(audio_vggish_pipeline(temp_path).get()[0])
# print(test_embed.shape)
# os.unlink(temp_path)
#
#
# test_id = 'null_c-45AfEdAU050_99000_109000'
# test_path = f'{data_dir}/media/{test_id}/audio.wav'
# temp_path = preprocess_audio_to_mono(test_path)
# print(f"original audio info: {torchaudio.info(test_path)}")
# print(f"mono audio info: :{torchaudio.info(temp_path)}")
# test_embed = torch.tensor(audio_vggish_pipeline(temp_path).get()[0])
# print(test_embed.shape)
# os.unlink(temp_path)



metapath = os.path.join(data_dir, 'metadata.csv')
metadata = pd.read_csv(metapath, header=0)
metadata = metadata[metadata['split'].isin(['train', 'val', 'test_s', 'test_u', 'test_n'])]
# metadata = metadata[metadata['split'].isin(['test_s'])]

vids = metadata['uid'].apply(lambda x: x.rsplit('_', 2)[0]).unique()

save_dir = os.path.join(data_dir, 'audio_embed')
os.makedirs(save_dir, exist_ok=True)

for vid in vids:
    audio_path = f'{data_dir}/media/{vid}/audio.wav'
    temp_path = preprocess_audio_to_mono(audio_path)
    audio_embed = torch.tensor(audio_vggish_pipeline(temp_path).get()[0])
    os.unlink(temp_path)
    # print(f"{vid}: {audio_embed.shape}")
    torch.save(audio_embed, f'{save_dir}/{vid}.pt')
    print(f'{vid} embedding saved {audio_embed.shape}')

