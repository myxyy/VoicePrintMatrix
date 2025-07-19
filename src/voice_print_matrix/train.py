import torch
import torch.nn as nn
from voice_print_matrix.jvs_batch_dataset import JVSBatchDataset
from voice_print_matrix.ae import AutoEncoder
from tqdm import tqdm

jvs_dataset = JVSBatchDataset()
#print(jvs_dataset[6257])
#print(len(jvs_dataset))

model = AutoEncoder().to('cuda')

optimizer = torch.optim.Adam(model.parameters())

batch_size = 1
num_epoch = 10
dataloader = torch.utils.data.DataLoader(jvs_dataset, batch_size = batch_size, shuffle=True)
 

criterion = nn.MSELoss()

for _ in range(num_epoch):
    pbar = tqdm(dataloader)
    for batch in pbar:
        waveform, label = batch
        waveform = waveform.unsqueeze(1).to('cuda')
        optimizer.zero_grad()
        loss = criterion(waveform, model(waveform))
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

torch.save(model.state_dict(), 'weight/autoencoder.pth')
