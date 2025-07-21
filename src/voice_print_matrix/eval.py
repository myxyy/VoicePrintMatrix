import torch
from  voice_print_matrix.ae import AutoEncoder
from voice_print_matrix.vpm_ae import VPMAutoEncoder
import librosa
import torchaudio

vpm_ae_state_dict = torch.load("resources/weight/vpm_ae.pt")
model_vpm_ae = VPMAutoEncoder(dim_token=2048, dim_content=512, dim_print=512, dim=1024, dim_hidden=2048, num_layers=8).to('cuda')
model_vpm_ae.load_state_dict(vpm_ae_state_dict)
model_vpm_ae.eval()

segment_length = 2048
sample_rate = 22050

zundamon_waveform, _ = librosa.load('resources/zundamon.wav', sr=sample_rate)
zundamon_waveform = zundamon_waveform[:(len(zundamon_waveform) // segment_length) * segment_length]
zundamon_waveform_tensor = torch.tensor(zundamon_waveform, dtype=torch.float32).reshape(-1, segment_length).to('cuda')
zundamon_length = zundamon_waveform_tensor.shape[0]

metan_waveform, _ = librosa.load('resources/metan.wav', sr=22050)
metan_waveform = metan_waveform[:(len(metan_waveform) // segment_length) * segment_length]
metan_waveform_tensor = torch.tensor(metan_waveform, dtype=torch.float32).reshape(-1, segment_length).to('cuda')
metan_length = metan_waveform_tensor.shape[0]

zundamon_print = model_vpm_ae.print_encoder(zundamon_waveform_tensor[None,:,:]).reshape(zundamon_length, -1).mean(dim=0)
metan_content = model_vpm_ae.content_encoder(metan_waveform_tensor[None,:,:]).reshape(metan_length, -1)

metan_zundamon_transformed_waveform = model_vpm_ae.decoder(torch.cat((metan_content[None,:,:], zundamon_print[None,None,:].expand(1, metan_length, -1)), dim=-1)).reshape(metan_length, -1)

metan_zundamon_transformed_waveform = metan_zundamon_transformed_waveform.reshape(1,-1).cpu().detach()
torchaudio.save(uri='resources/metan_zundamon_transformed.wav', src=metan_zundamon_transformed_waveform, sample_rate=sample_rate, encoding="PCM_F")
