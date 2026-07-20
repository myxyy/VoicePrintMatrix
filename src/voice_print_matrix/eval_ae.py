import torch
from  voice_print_matrix.vpm_ae import AutoEncoder
import librosa
import torchaudio
from voice_print_matrix.config import RESOURCES_DIR

ae_state_dict = torch.load(RESOURCES_DIR / 'weight' / 'ae.pt')
model_ae = AutoEncoder(decoder_type='hifigan').to('cuda')
model_ae.load_state_dict(ae_state_dict)
model_ae.eval()

segment_length = 2048
sample_rate = 22050

zundamon_waveform, _ = librosa.load(RESOURCES_DIR / 'zundamon.wav', sr=sample_rate)
zundamon_waveform = zundamon_waveform[:(len(zundamon_waveform) // segment_length) * segment_length]
zundamon_waveform_tensor = torch.tensor(zundamon_waveform, dtype=torch.float32).reshape(-1, segment_length).to('cuda')

zundamon_reconstructed, _ = model_ae(zundamon_waveform_tensor[None,:,:])
zundamon_reconstructed = zundamon_reconstructed.reshape(1,-1).cpu().detach()

torchaudio.save(uri=str(RESOURCES_DIR / 'zundamon_reconstructed.wav'), src=zundamon_reconstructed, sample_rate=sample_rate, encoding="PCM_F")