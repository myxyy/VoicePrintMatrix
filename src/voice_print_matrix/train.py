import torch
import torch.nn as nn
from voice_print_matrix.jvs_batch_dataset import JVSBatchDataset
from voice_print_matrix.ae import AutoEncoder
from voice_print_matrix.vpm_ae import VPMAutoEncoder
from tqdm import tqdm

jvs_dataset = JVSBatchDataset()
#print(jvs_dataset[6257])
#print(len(jvs_dataset))

model_ae = AutoEncoder().to('cuda')
model_vpm_ae = VPMAutoEncoder(dim_token=512, dim_content=256, dim_print=256, dim=512, dim_hidden=1024, num_layers=8).to('cuda')
model_ae.train()
model_vpm_ae.train()

optimizer_ae = torch.optim.AdamW(model_ae.parameters(), lr=1e-5)
optimizer_vpm_ae = torch.optim.AdamW(model_vpm_ae.parameters(), lr=1e-5)

batch_size = 8
num_epoch = 10
dataloader = torch.utils.data.DataLoader(jvs_dataset, batch_size = batch_size, shuffle=True)
 

criterion = nn.MSELoss()

for _ in range(num_epoch):
    pbar = tqdm(dataloader)
    for batch in pbar:
        waveform, label = batch
        batch_size, length, segment_length = waveform.shape
        waveform = waveform.reshape(batch_size * length, 1, segment_length).to('cuda')
        optimizer_ae.zero_grad()
        optimizer_vpm_ae.zero_grad()
        waveform_reconstructed, latent = model_ae(waveform)
        latent = latent.reshape(batch_size, length, -1)
        loss_ae = criterion(waveform, waveform_reconstructed)

        latent_reconstructed, content, voice_print = model_vpm_ae(latent)
        loss_vpm_ae = criterion(latent, latent_reconstructed)

        voice_print_matrix = nn.functional.cosine_similarity(voice_print[:,:,None,:], voice_print[:,None,:,:])
        voice_print_matrix_coef = torch.where(label[:, :, None] == label[:, None, :], 1.0, -1.0).triu(1).to(voice_print_matrix.device)
        voice_print_matrix *= voice_print_matrix_coef
        loss_vp = torch.mean(voice_print_matrix)

        loss = loss_ae + loss_vpm_ae + loss_vp
        loss.backward()
        optimizer_ae.step()
        optimizer_vpm_ae.step()
        pbar.set_postfix(loss_ae=loss_ae.item(), loss_vpm_ae=loss_vpm_ae.item(), loss_vp=loss_vp.item(), loss=loss.item())

torch.save(model_ae.state_dict(), 'resources/weight/ae.pt')
torch.save(model_vpm_ae.state_dict(), 'resources/weight/vpm_ae.pt')
