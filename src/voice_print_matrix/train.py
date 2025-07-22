import torch
import torch.nn as nn
from voice_print_matrix.jvs_batch_dataset import JVSBatchDataset
from voice_print_matrix.ae import AutoEncoder
from voice_print_matrix.vpm_ae import VPMAutoEncoder
from tqdm import tqdm
import os
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from voice_print_matrix.utils import multiscale_spectrum

gpu_id = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
torch.cuda.set_device(gpu_id)
init_process_group(backend="nccl")

segment_per_batch = 256

jvs_dataset = JVSBatchDataset(segments_per_batch=segment_per_batch, size_ratio=1)
#print(jvs_dataset[6257])
#print(len(jvs_dataset))


model_vpm_ae = VPMAutoEncoder(waveform_length=2048, dim_content=512, dim_print=512, dim=1024, dim_hidden=2048, num_layers=8).to('cuda')
model_vpm_ae.train()
model_vpm_ae = DDP(model_vpm_ae, device_ids=[gpu_id])

batch_size = 4
lr = 1e-5

optimizer_vpm_ae = torch.optim.AdamW(model_vpm_ae.parameters(), lr=lr)

num_epoch = 10


criterion = nn.MSELoss()

voice_print_permutation_split = 8

sampler = DistributedSampler(jvs_dataset, shuffle=True, num_replicas=world_size, rank=gpu_id, drop_last=True)
dataloader = torch.utils.data.DataLoader(jvs_dataset, batch_size = batch_size, pin_memory=True, shuffle=False, sampler=sampler)

for epoch in range(num_epoch):
    pbar = tqdm(dataloader, disable=(gpu_id != 0))
    for batch in pbar:
        waveform, label = batch
        batch_size, length, segment_length = waveform.shape

        optimizer_vpm_ae.zero_grad()

        assert length % voice_print_permutation_split == 0, "Length must be divisible by voice_print_permutation_split"
        waveform = waveform.to('cuda')

        waveform_reconstructed, content, voice_print = model_vpm_ae(waveform)
        #waveform_spectrum = multiscale_spectrum(waveform.reshape(batch_size * length, -1))
        #waveform_reconstructed_spectrum = multiscale_spectrum(waveform_reconstructed.reshape(batch_size * length, -1))
        #loss_ae = criterion(waveform_spectrum, waveform_reconstructed_spectrum)
        loss_ae = criterion(waveform, waveform_reconstructed)

        voice_print_matrix = nn.functional.cosine_similarity(voice_print[:,:,None,:], voice_print[:,None,:,:],dim=-1)
        voice_print_matrix_coef = torch.where(label[:, :, None] == label[:, None, :], 1.0, -1.0).triu(1).to(voice_print_matrix.device)
        voice_print_matrix *= voice_print_matrix_coef
        loss_vp = -voice_print_matrix.sum() / torch.ones_like(voice_print_matrix, device=voice_print_matrix.device).triu(1).sum()

        voice_print_permuted = voice_print.reshape(batch_size, voice_print_permutation_split, length // voice_print_permutation_split, voice_print.shape[-1])
        voice_print_permuted = voice_print_permuted[:, torch.randperm(voice_print_permutation_split), :, :]
        voice_print_permuted = voice_print_permuted.reshape(batch_size, length, voice_print.shape[-1])
        with torch.no_grad():
            permuted_reconstructed_waveform = model_vpm_ae.module.decoder(torch.cat((content, voice_print_permuted), dim=-1))
        content_reconstructed = model_vpm_ae.module.content_encoder(permuted_reconstructed_waveform)
        voice_print_permuted_reconstructed = model_vpm_ae.module.print_encoder(permuted_reconstructed_waveform)
        loss_udc = criterion(content.detach(), content_reconstructed)
        loss_udp = criterion(voice_print_permuted.detach(), voice_print_permuted_reconstructed)

        loss = loss_ae + loss_vp + loss_udc + loss_udp
        loss.backward()
        #loss_ae = dist.all_reduce(loss_ae, op=dist.ReduceOp.AVG)
        #loss_vpm_ae = dist.all_reduce(loss_vpm_ae, op=dist.ReduceOp.AVG)
        #loss_vp = dist.all_reduce(loss_vp, op=dist.ReduceOp.AVG)
        #loss_udc = dist.all_reduce(loss_udc, op=dist.ReduceOp.AVG)
        #loss_udp = dist.all_reduce(loss_udp, op=dist.ReduceOp.AVG)
        optimizer_vpm_ae.step()
        if gpu_id == 0:
            pbar.set_postfix(loss_ae=loss_ae.item(), loss_vp=loss_vp.item(), loss_udc=loss_udc.item(), loss_udp=loss_udp.item())

    if gpu_id == 0:
        print("Saving model weights...")
        torch.save(model_vpm_ae.module.state_dict(), 'resources/weight/vpm_ae.pt')

destroy_process_group()
