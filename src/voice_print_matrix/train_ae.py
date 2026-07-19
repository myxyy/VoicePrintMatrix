import torch
import torch.nn as nn
from voice_print_matrix.jvs_batch_dataset import JVSBatchDataset
from voice_print_matrix.vpm_ae import AutoEncoder
from tqdm import tqdm
from voice_print_matrix.utils import MultiResolutionSTFTLoss, multiscale_spectrum
from voice_print_matrix.config import RESOURCES_DIR
import os

loss_type = 'mrstft'  # 'mrstft': multi-resolution STFT損失 / 'multiscale': 旧損失(セグメント単位のmultiscale spectrum MSE)

segment_per_batch = 256

jvs_dataset = JVSBatchDataset(segments_per_batch=segment_per_batch, size_ratio=1)
#print(jvs_dataset[6257])
#print(len(jvs_dataset))

model_ae = AutoEncoder(decoder_type='hifigan').to('cuda')
model_ae.train()

optimizer_ae = torch.optim.AdamW(model_ae.parameters(), lr=1e-4)

batch_size = 4
num_epoch = 20
dataloader = torch.utils.data.DataLoader(jvs_dataset, batch_size = batch_size, shuffle=True)
 

criterion_mrstft = MultiResolutionSTFTLoss().to('cuda')
criterion_mse = nn.MSELoss()


for i in range(num_epoch):
    pbar = tqdm(dataloader, desc=f"Epoch {i+1}/{num_epoch}")
    for batch in pbar:
        waveform, label = batch
        optimizer_ae.zero_grad()
        batch_size, length, segment_length = waveform.shape
        waveform = waveform.to('cuda')
        waveform_reconstructed, latent = model_ae(waveform)
        if loss_type == 'mrstft':
            # セグメントを連結した波形全体で損失をとり、境界の不連続も損失に反映させる
            loss = criterion_mrstft(waveform_reconstructed.reshape(batch_size, length * segment_length), waveform.reshape(batch_size, length * segment_length))
        else:
            min_length = 64
            waveform_spectrum = torch.log1p(multiscale_spectrum(waveform.reshape(batch_size * length, segment_length), min_length=min_length))
            waveform_reconstructed_spectrum = torch.log1p(multiscale_spectrum(waveform_reconstructed.reshape(batch_size * length, segment_length), min_length=min_length))
            loss = criterion_mse(waveform_spectrum, waveform_reconstructed_spectrum)
        loss.backward()
        optimizer_ae.step()
        pbar.set_postfix(loss=loss.item())

    print("Saving model weights...")
    os.makedirs(RESOURCES_DIR / 'weight', exist_ok=True)
    torch.save(model_ae.state_dict(), RESOURCES_DIR / 'weight' / 'ae.pt')

