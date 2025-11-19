import torch
import torch.nn as nn
from voice_print_matrix.jvs_batch_dataset import JVSBatchDataset
from voice_print_matrix.vpm_ae import AutoEncoder
from tqdm import tqdm
from voice_print_matrix.utils import multiscale_spectrum
import os

segment_per_batch = 256

jvs_dataset = JVSBatchDataset(segments_per_batch=segment_per_batch, size_ratio=1)
#print(jvs_dataset[6257])
#print(len(jvs_dataset))

model_ae = AutoEncoder().to('cuda')
model_ae.train()

optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=1e-5)

batch_size = 4
num_epoch = 20
dataloader = torch.utils.data.DataLoader(jvs_dataset, batch_size = batch_size, shuffle=True)
 

criterion = nn.MSELoss()


for i in range(num_epoch):
    pbar = tqdm(dataloader, desc=f"Epoch {i+1}/{num_epoch}")
    for batch in pbar:
        waveform, label = batch
        optimizer_ae.zero_grad()
        batch_size, length, segment_length = waveform.shape
        waveform = waveform.to('cuda')
        waveform_reconstructed, latent = model_ae(waveform)
        min_length = 64
        waveform_spectrum = multiscale_spectrum(waveform.reshape(batch_size * length, segment_length), min_length=min_length)
        waveform_reconstructed_spectrum = multiscale_spectrum(waveform_reconstructed.reshape(batch_size * length, segment_length), min_length=min_length)
        loss = criterion(waveform_spectrum, waveform_reconstructed_spectrum)
        loss.backward()
        optimizer_ae.step()
        pbar.set_postfix(loss=loss.item())

    print("Saving model weights...")
    if not os.path.exists('resources/weight'):
        os.makedirs('resources/weight')
    torch.save(model_ae.state_dict(), 'resources/weight/ae.pt')

