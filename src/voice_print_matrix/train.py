import torch
import torch.nn as nn
from voice_print_matrix.jvs_batch_dataset import JVSBatchDataset
from voice_print_matrix.ae import AutoEncoder
from tqdm import tqdm

jvs_dataset = JVSBatchDataset()
#print(jvs_dataset[6257])
#print(len(jvs_dataset))

model = AutoEncoder().to('cuda')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

batch_size = 1
num_epoch = 10
dataloader = torch.utils.data.DataLoader(jvs_dataset, batch_size = batch_size, shuffle=True)
 

criterion = nn.MSELoss()

for _ in range(num_epoch):
    pbar = tqdm(dataloader)
    for batch in pbar:
        waveform, label = batch
        batch_size, length, segment_length = waveform.shape
        waveform = waveform.reshape(batch_size * length, 1, segment_length).to('cuda')
        optimizer.zero_grad()
        loss = criterion(waveform, model(waveform))
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

torch.save(model.state_dict(), 'weight/autoencoder.pth')
