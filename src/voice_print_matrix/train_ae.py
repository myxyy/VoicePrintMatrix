import torch
import torch.nn as nn
from voice_print_matrix.jvs_batch_dataset import JVSBatchDataset
from voice_print_matrix.vpm_ae import AutoEncoder
from tqdm import tqdm
from voice_print_matrix.utils import MultiResolutionSTFTLoss, multiscale_spectrum
from voice_print_matrix.config import RESOURCES_DIR
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

loss_type = 'mrstft'  # 'mrstft': multi-resolution STFT損失 / 'multiscale': 旧損失(セグメント単位のmultiscale spectrum MSE)

gpu_id = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
torch.cuda.set_device(gpu_id)
init_process_group(backend="nccl")

segment_per_batch = 256

jvs_dataset = JVSBatchDataset(segments_per_batch=segment_per_batch, size_ratio=1)
#print(jvs_dataset[6257])
#print(len(jvs_dataset))

model_ae = AutoEncoder(decoder_type='hifigan').to(gpu_id)
model_ae.train()
model_ae = DDP(model_ae, device_ids=[gpu_id])

batch_size = 4  # 1GPUあたり。実効バッチサイズは batch_size * world_size

optimizer_ae = torch.optim.AdamW(model_ae.parameters(), lr=2.5e-5 * batch_size * world_size)

num_epoch = 20
sampler = DistributedSampler(jvs_dataset, num_replicas=world_size, rank=gpu_id, shuffle=True, drop_last=True)
dataloader = torch.utils.data.DataLoader(jvs_dataset, batch_size = batch_size, pin_memory=True, shuffle=False, sampler=sampler)

criterion_mrstft = MultiResolutionSTFTLoss().to(gpu_id)
criterion_mse = nn.MSELoss()


for i in range(num_epoch):
    sampler.set_epoch(i)
    pbar = tqdm(dataloader, desc=f"Epoch {i+1}/{num_epoch}", disable=(gpu_id != 0))
    for batch in pbar:
        waveform, label = batch
        optimizer_ae.zero_grad()
        batch_size, length, segment_length = waveform.shape
        waveform = waveform.to(gpu_id)
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
        if gpu_id == 0:
            pbar.set_postfix(loss=loss.item())

    if gpu_id == 0:
        print("Saving model weights...")
        os.makedirs(RESOURCES_DIR / 'weight', exist_ok=True)
        torch.save(model_ae.module.state_dict(), RESOURCES_DIR / 'weight' / 'ae.pt')

destroy_process_group()

