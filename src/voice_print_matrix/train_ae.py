import torch
import torch.nn as nn
from voice_print_matrix.jvs_batch_dataset import JVSBatchDataset
from voice_print_matrix.vpm_ae import AutoEncoder
from tqdm import tqdm
from voice_print_matrix.utils import multiscale_spectrum

segment_per_batch = 256

jvs_dataset = JVSBatchDataset(segments_per_batch=segment_per_batch, size_ratio=1)
#print(jvs_dataset[6257])
#print(len(jvs_dataset))

model_ae = AutoEncoder().to('cuda')
model_ae.train()

optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=1e-5)

batch_size = 8
num_epoch = 10
dataloader = torch.utils.data.DataLoader(jvs_dataset, batch_size = batch_size, shuffle=True)
 

criterion = nn.MSELoss()

voice_print_permutation_split = 8

for _ in range(num_epoch):
    pbar = tqdm(dataloader)
    for batch in pbar:
        waveform, label = batch
        optimizer_ae.zero_grad()
        batch_size, length, segment_length = waveform.shape
        assert length % voice_print_permutation_split == 0, "Length must be divisible by voice_print_permutation_split"
        waveform = waveform.reshape(batch_size * length, 1, segment_length).to('cuda')
        waveform_reconstructed, latent = model_ae(waveform)
        waveform_spectrum = multiscale_spectrum(waveform.squeeze(1))
        waveform_reconstructed_spectrum = multiscale_spectrum(waveform_reconstructed.squeeze(1))
        loss = criterion(waveform_spectrum, waveform_reconstructed_spectrum)
        loss.backward()
        optimizer_ae.step()
        pbar.set_postfix(loss=loss.item())

    print("Saving model weights...")
    torch.save(model_ae.state_dict(), 'resources/weight/ae.pt')

