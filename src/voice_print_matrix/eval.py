import torch
from  voice_print_matrix.ae import AutoEncoder
from voice_print_matrix.vpm_ae import VPMAutoEncoder
import librosa
import torchaudio

ae_state_dict = torch.load("resources/weight/ae.pt")
vpm_ae_state_dict = torch.load("resources/weight/vpm_ae.pt")
model_ae = AutoEncoder().to('cuda')
model_vpm_ae = VPMAutoEncoder(dim_token=512, dim_content=256, dim_print=256, dim=512, dim_hidden=1024, num_layers=8).to('cuda')
model_ae.load_state_dict(ae_state_dict)
model_vpm_ae.load_state_dict(vpm_ae_state_dict)

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

zundamon_tokens = model_ae.encoder(zundamon_waveform_tensor[:,None,:])
zundamon_tokens = zundamon_tokens.reshape(zundamon_length, -1)
metan_tokens = model_ae.encoder(metan_waveform_tensor[:,None,:])
metan_tokens_shape = metan_tokens.shape
metan_tokens = metan_tokens.reshape(metan_length, -1)

zundamon_print = model_vpm_ae.print_encoder(zundamon_tokens[None,:,:]).reshape(zundamon_length, -1).mean(dim=0)
metan_content = model_vpm_ae.content_encoder(metan_tokens[None,:,:]).reshape(metan_length, -1)

metan_zundamon_transformed_tokens = model_vpm_ae.decoder(torch.cat((metan_content[None,:,:], zundamon_print[None,None,:].expand(1, metan_length, -1)), dim=-1)).reshape(metan_length, -1)
metan_zundamon_transformed_tokens = metan_zundamon_transformed_tokens.reshape(metan_tokens_shape)

metan_zundamon_transformed = model_ae.decoder(metan_zundamon_transformed_tokens)
metan_zundamon_transformed = metan_zundamon_transformed.reshape(1,-1).cpu().detach()
torchaudio.save(uri='resources/metan_zundamon_transformed.wav', src=metan_zundamon_transformed, sample_rate=sample_rate, encoding="PCM_F")

metan_reconstructed = model_ae.decoder(metan_tokens.reshape(metan_tokens_shape)).reshape(1,-1).cpu().detach()
torchaudio.save(uri='resources/metan_reconstructed.wav', src=metan_reconstructed, sample_rate=sample_rate, encoding="PCM_F")