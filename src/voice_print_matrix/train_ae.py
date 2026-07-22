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

from voice_print_matrix.discriminator import HiFiGANDiscriminator, discriminator_loss, generator_adversarial_loss, feature_matching_loss

loss_type = 'mrstft'  # 'mrstft': multi-resolution STFT損失 / 'multiscale': 旧損失(セグメント単位のmultiscale spectrum MSE)

use_gan = True
lambda_spec = 10.0  # スペクトル損失の重み(HiFi-GANのmel重み45相当。mrstftはmel L1よりスケールが大きいため小さめ)
lambda_adv = 1.0
lambda_fm = 2.0
# 敵対的損失は全系列(約95秒/GPU)ではなくランダムな短いクロップにのみ適用する。
# 本家HiFi-GANも約0.4秒クロップで学習しており、これでDiscriminatorのVRAMを小さく抑える
gan_crop_length = 16384
gan_crops_per_item = 2

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

# 1GPUあたり。実効バッチサイズは batch_size * world_size。
# GAN有効時はDiscriminatorのパラメータ・optimizer状態・クロップ活性化の分(約3GB)を空けるため3に落とす(24GB GPUでピーク約19GB)
batch_size = 3 if use_gan else 4

optimizer_ae = torch.optim.AdamW(model_ae.parameters(), lr=2.5e-5 * batch_size * world_size)

if use_gan:
    model_disc = HiFiGANDiscriminator().to(gpu_id)
    model_disc.train()
    model_disc = DDP(model_disc, device_ids=[gpu_id])
    optimizer_disc = torch.optim.AdamW(model_disc.parameters(), lr=2.5e-5 * batch_size * world_size, betas=(0.8, 0.99))

num_epoch = 20
sampler = DistributedSampler(jvs_dataset, num_replicas=world_size, rank=gpu_id, shuffle=True, drop_last=True)
dataloader = torch.utils.data.DataLoader(jvs_dataset, batch_size = batch_size, pin_memory=True, shuffle=False, sampler=sampler)

criterion_mrstft = MultiResolutionSTFTLoss().to(gpu_id)
criterion_mse = nn.MSELoss()


for i in range(num_epoch):
    sampler.set_epoch(i)
    pbar = tqdm(dataloader, desc=f"Epoch {i+1}/{num_epoch}", disable=(gpu_id != 0), dynamic_ncols=True)
    for batch in pbar:
        waveform, label = batch
        batch_size, length, segment_length = waveform.shape
        waveform = waveform.to(gpu_id)
        waveform_reconstructed, latent = model_ae(waveform)
        waveform_flat = waveform.reshape(batch_size, length * segment_length)
        waveform_reconstructed_flat = waveform_reconstructed.reshape(batch_size, length * segment_length)

        if use_gan:
            # realとfakeで同一位置のランダムクロップを切り出す
            total_length = waveform_flat.shape[1]
            crop_slices = [(b, s) for b in range(batch_size) for s in torch.randint(0, total_length - gan_crop_length + 1, (gan_crops_per_item,)).tolist()]
            real_crop = torch.stack([waveform_flat[b, s:s + gan_crop_length] for b, s in crop_slices])[:, None, :]
            fake_crop = torch.stack([waveform_reconstructed_flat[b, s:s + gan_crop_length] for b, s in crop_slices])[:, None, :]

            # Discriminator更新(fakeはdetachしてGeneratorへの勾配を止める)
            optimizer_disc.zero_grad()
            logits_real, logits_fake, _, _ = model_disc(real_crop, fake_crop.detach())
            loss_disc = discriminator_loss(logits_real, logits_fake)
            loss_disc.backward()
            optimizer_disc.step()

        # Generator更新
        optimizer_ae.zero_grad()
        if loss_type == 'mrstft':
            # セグメントを連結した波形全体で損失をとり、境界の不連続も損失に反映させる
            loss_spec = criterion_mrstft(waveform_reconstructed_flat, waveform_flat)
        else:
            min_length = 64
            waveform_spectrum = torch.log1p(multiscale_spectrum(waveform_flat.reshape(batch_size * length, segment_length), min_length=min_length))
            waveform_reconstructed_spectrum = torch.log1p(multiscale_spectrum(waveform_reconstructed_flat.reshape(batch_size * length, segment_length), min_length=min_length))
            loss_spec = criterion_mse(waveform_spectrum, waveform_reconstructed_spectrum)
        if use_gan:
            logits_real, logits_fake, features_real, features_fake = model_disc(real_crop, fake_crop)
            loss_adv = generator_adversarial_loss(logits_fake)
            loss_fm = feature_matching_loss(features_real, features_fake)
            loss = lambda_spec * loss_spec + lambda_adv * loss_adv + lambda_fm * loss_fm
        else:
            loss = loss_spec
        loss.backward()
        optimizer_ae.step()
        if gpu_id == 0:
            if use_gan:
                pbar.set_postfix(loss_spec=loss_spec.item(), loss_adv=loss_adv.item(), loss_fm=loss_fm.item(), loss_disc=loss_disc.item())
            else:
                pbar.set_postfix(loss=loss_spec.item())

    if gpu_id == 0:
        print("Saving model weights...")
        os.makedirs(RESOURCES_DIR / 'weight', exist_ok=True)
        torch.save(model_ae.module.state_dict(), RESOURCES_DIR / 'weight' / 'ae.pt')
        if use_gan:
            torch.save(model_disc.module.state_dict(), RESOURCES_DIR / 'weight' / 'disc_ae.pt')

destroy_process_group()

