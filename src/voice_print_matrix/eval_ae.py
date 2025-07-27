import torch
from  voice_print_matrix.vpm_ae import AutoEncoder
import librosa
import torchaudio

ae_state_dict = torch.load("resources/weight/ae.pt")
model_ae = AutoEncoder().to('cuda')
model_ae.load_state_dict(ae_state_dict)
model_ae.eval()

segment_length = 2048
sample_rate = 22050

zundamon_waveform, _ = librosa.load('resources/zundamon.wav', sr=sample_rate)
zundamon_waveform = zundamon_waveform[:(len(zundamon_waveform) // segment_length) * segment_length]
zundamon_waveform_tensor = torch.tensor(zundamon_waveform, dtype=torch.float32).reshape(-1, segment_length).to('cuda')

zundamon_reconstructed, _ = model_ae(zundamon_waveform_tensor[None,:,:])
zundamon_reconstructed = zundamon_reconstructed.reshape(1,-1).cpu().detach()

torchaudio.save(uri='resources/zundamon_reconstructed.wav', src=zundamon_reconstructed, sample_rate=sample_rate, encoding="PCM_F")