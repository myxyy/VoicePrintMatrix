from voice_print_matrix.qgru import QGRUModel
import torch.nn as nn
import torch
import torchaudio
import numpy as np
import math
from torch.nn.utils.parametrizations import weight_norm

class Encoder(nn.Module):
    def __init__(self, waveform_length=2048, dim=512, dim_hidden=2048, num_layers=4, n_mels=128, n_fft=1024, num_latent_frames=1):
        super().__init__()
        sample_rate = 22050
        # n_melsを増やす場合はn_fftも上げること: FFTビン間隔(sr/n_fft)より低域melフィルタの
        # 帯域幅が狭くなると空フィルタ(常にゼロの特徴)が生じる。n_mels=128にはn_fft>=1024が必要
        hop_length = 256
        self.num_latent_frames = num_latent_frames
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, f_max=sample_rate // 2, hop_length=hop_length, n_mels=n_mels, center=False)
        num_steps = (waveform_length - n_fft) // hop_length + 1
        self.qgru = QGRUModel(dim_in=n_mels * num_steps, dim_out=dim * num_latent_frames, dim=dim, dim_hidden=dim_hidden, num_layers=num_layers)

    def forward(self, x):
        # 出力はセグメントあたり num_latent_frames 本の潜在ベクトル列 (batch, length * num_latent_frames, dim)
        batch, length, _ = x.shape
        mel_spectrogram = self.transform(x)
        x = mel_spectrogram.reshape(batch, length, -1)
        x = self.qgru(x)
        x = x.reshape(batch, length * self.num_latent_frames, -1)
        return x

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=2048):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, waveform_length=2048, dim=512, dim_hidden=2048, num_layers=4, num_oscillators=32):
        super().__init__()
        self.num_oscillators = num_oscillators
        self.waveform_length = waveform_length
        self.qgru = QGRUModel(dim_in=dim, dim_out=dim, dim=dim, dim_hidden=dim_hidden, num_layers=num_layers)
        self.phase_offset = nn.Parameter(torch.zeros(num_oscillators))
        self.fs_fc = nn.Linear(dim, waveform_length)
        self.log_fs_scale = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.fs_fc.weight)
        nn.init.zeros_(self.fs_fc.bias)
        #self.fs_fc = MLP(dim, waveform_length, dim_hidden=dim_hidden)
        #self.amp_fc = nn.Parameter(torch.randn(num_oscillators, dim, waveform_length) * dim ** -0.5)
        self.amp_fc = nn.Linear(dim, num_oscillators)
        self.log_amp_temperature = nn.Parameter(torch.zeros(1))
        self.amp_whole_fc = nn.Linear(dim, waveform_length)
        self.log_amp_whole_temperature = nn.Parameter(torch.zeros(1))

        #self.norm = nn.LayerNorm(dim)
        #self.fc_noise_1 = nn.Linear(dim + waveform_length, dim_hidden + waveform_length)
        #self.fc_noise_2 = nn.Linear(dim_hidden + waveform_length, waveform_length)

        #self.amp_filter = nn.Parameter(torch.randn(dim, num_oscillators, waveform_length * 2) * dim ** -0.5)
        #self.amp_whole_filter = nn.Linear(dim, waveform_length * 2)
        #self.fs_filter = nn.Linear(dim, waveform_length * 2)
        #self.log_fs_filter_temperature = nn.Parameter(torch.zeros(1))
        self.waveform_filter = nn.Parameter(torch.randn(dim, num_oscillators, waveform_length) * dim ** -0.5)
        self.log_waveform_filter_temperature = nn.Parameter(torch.zeros(num_oscillators))
        #self.act = nn.SiLU()
        #nn.init.zeros_(self.fc_log_amp_init.weight)
        #nn.init.zeros_(self.fc_log_amp_init.bias)

    def forward(self, x):
        batch, length, dim = x.shape
        x = self.qgru(x)
        x = x.reshape(batch * length, dim)

        fs = self.fs_fc(x)

        #fs_fft = torch.fft.rfft(nn.functional.pad(fs, (0, self.waveform_length), "constant", 0), dim=-1)
        #fs_filter_temperature = torch.exp(self.log_fs_filter_temperature)
        #fs_filter = torch.softmax(self.fs_filter(x) * fs_filter_temperature, dim=-1)
        #fs_filter_fft = torch.fft.rfft(fs_filter, dim=-1)
        #fs = torch.fft.irfft(fs_fft * fs_filter_fft, n=self.waveform_length, dim=-1)

        #fs = torch.sigmoid(fs) * torch.exp(self.log_fs_scale)
        fs = torch.sigmoid(fs)
        # 位相はセグメント内ではなく全セグメントを連結した時間軸で累積し、境界の不連続をなくす。
        # 長時間の累積はfloat32では精度が足りなくなるため、float64で累積し
        # 全倍音に共通の周期 2*num_oscillators で剰余をとってからfloat32に戻す
        fs = fs.reshape(batch, length * self.waveform_length).double()
        fs = torch.cumsum(fs, dim=-1) % (2 * self.num_oscillators)
        fs = fs.float().reshape(batch * length, self.waveform_length)
        fs = fs[:,None,:] * (torch.arange(self.num_oscillators, device=x.device) + 1)[None,:,None]

        #amp = torch.einsum("bd, odw -> bow", x, self.amp_fc)
        amp = self.amp_fc(x)
        #amp = -self.act(amp)
        #amp[:,0,:] = 0  # Set the first oscillator's amplitude to zero
        amp_temperature = torch.exp(self.log_amp_temperature)
        amp = torch.softmax(amp * amp_temperature, dim=1)

        #amp_fft = torch.fft.rfft(nn.functional.pad(amp, (0, self.waveform_length), "constant", 0), dim=-1)
        #amp_filter = torch.softmax(torch.einsum("bd, dow -> bow", x, self.amp_filter), dim=-1)
        #amp_filter_fft = torch.fft.rfft(amp_filter, dim=-1)
        #amp = torch.fft.irfft(amp_fft * amp_filter_fft, n=self.waveform_length, dim=-1)

        amp_whole_temperature = torch.exp(self.log_amp_whole_temperature)
        amp_whole = torch.exp(self.amp_whole_fc(x) * amp_whole_temperature)

        #amp_whole_fft = torch.fft.rfft(nn.functional.pad(amp_whole, (0, self.waveform_length), "constant", 0), dim=-1)
        #amp_whole_filter = torch.softmax(self.amp_whole_filter(x), dim=-1)
        #amp_whole_filter_fft = torch.fft.rfft(amp_whole_filter, dim=-1)
        #amp_whole = torch.fft.irfft(amp_whole_fft * amp_whole_filter_fft, n=self.waveform_length, dim=-1)

        arg = fs / self.num_oscillators + self.phase_offset[None,:,None]
        base_wave = torch.sin(arg * torch.pi)

        waveform = amp[:,:,None] * base_wave
        #waveform = base_wave
        waveform = amp_whole[:,None,:] * waveform # (batch * length, num_oscillators, waveform_length)

        waveform_fft = torch.fft.rfft(nn.functional.pad(waveform, (0, self.waveform_length), "constant", 0), dim=-1)
        waveform_filter_temperature = torch.exp(self.log_waveform_filter_temperature)
        waveform_filter = torch.einsum("bd, dow -> bow", x, self.waveform_filter)
        waveform_filter = torch.softmax(waveform_filter * waveform_filter_temperature[None,:,None], dim=-1)
        waveform_filter_fft = torch.fft.rfft(nn.functional.pad(waveform_filter, (0, self.waveform_length), "constant", 0), dim=-1)
        # FFTサイズ 2*waveform_length で線形畳み込みを復元する
        # (旧実装の irfft(n=waveform_length) は周波数ビンの切り詰めにより信号が2倍にデシメーションされていた)
        waveform = torch.fft.irfft(waveform_fft * waveform_filter_fft, n=self.waveform_length * 2, dim=-1)

        waveform = waveform.sum(dim=1)

        # フィルタの尾部がセグメントからはみ出した分は次のセグメントへオーバーラップアド
        # (最終セグメントの尾部はバッチ範囲外のため破棄)
        waveform = waveform.reshape(batch, length, 2 * self.waveform_length)
        head = waveform[:, :, :self.waveform_length]
        tail = waveform[:, :, self.waveform_length:]
        waveform = head + nn.functional.pad(tail[:, :-1, :], (0, 0, 1, 0))
        return waveform

class HiFiGANConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for dilation in dilations:
            conv = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=((kernel_size - 1) // 2) * dilation, dilation=dilation))
            self.convs.append(conv)
        self.act = nn.SiLU()

    def forward(self, x):
        for conv in self.convs:
            x = self.act(x)
            x = conv(x)
        return x

class HiFiGANResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations_list):
        super().__init__()
        self.blocks = nn.ModuleList()
        for dilations in dilations_list:
            block = HiFiGANConvBlock(channels, kernel_size, dilations)
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x_res = x
            x = block(x)
            x = x + x_res
        return x

class HiFiGANMRFBlock(nn.Module):
    def __init__(self, channels, kernel_sizes, dilations_list_list):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for kernel_size, dilations_list in zip(kernel_sizes, dilations_list_list):
            res_block = HiFiGANResBlock(channels, kernel_size, dilations_list)
            self.res_blocks.append(res_block)

    def forward(self, x):
        outputs = None
        for res_block in self.res_blocks:
            if outputs is None:
                outputs = res_block(x)
            else:
                outputs = outputs + res_block(x)
        return outputs

class HiFiGANUpsampleBlock(nn.Module):
    def __init__(self, channels, stride, kernel_sizes, dilations_list_list):
        super().__init__()
        self.mrf_block = HiFiGANMRFBlock(channels // 2, kernel_sizes, dilations_list_list)
        # kernel=2*stride, padding=stride/2 でちょうどstride倍のアップサンプルになり、
        # カーネルがストライドで割り切れるためチェッカーボードアーティファクトが出ない
        self.conv_transpose = weight_norm(nn.ConvTranspose1d(channels, channels // 2, kernel_size=stride * 2, stride=stride, padding=stride // 2))
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(x)
        x = self.conv_transpose(x)
        x = self.mrf_block(x)
        return x

class HiFiGANDecoder(nn.Module):
    def __init__(self, waveform_length=2048, dim=512, dim_hidden=2048, num_layers=4, upsample_rates=[8, 8, 2, 2], initial_channel=256, kernel_sizes=[3, 7, 11], dilations_list_list=[[[1, 1], [3, 1], [5, 1]], [[1, 1], [3, 1], [5, 1]], [[1, 1], [3, 1], [5, 1]]]):
        super().__init__()
        self.qgru = QGRUModel(dim_in=dim, dim_out=dim, dim=dim, dim_hidden=dim_hidden, num_layers=num_layers)
        total_upsample = math.prod(upsample_rates)
        assert waveform_length % total_upsample == 0, "waveform_length must be divisible by prod(upsample_rates)"
        initial_length = waveform_length // total_upsample
        self.fc = nn.Linear(dim, initial_channel * initial_length)
        self.upsample_blocks = nn.ModuleList()
        self.initial_channel = initial_channel
        channel = initial_channel
        for rate in upsample_rates:
            upsample_block = HiFiGANUpsampleBlock(channel, rate, kernel_sizes, dilations_list_list)
            self.upsample_blocks.append(upsample_block)
            channel = channel // 2
        self.conv_final = weight_norm(nn.Conv1d(channel, 1, kernel_size=7, padding=3))
        self.act = nn.SiLU()
        self.act_final = nn.Tanh()
    
    def forward(self, x):
        batch, length, dim = x.shape
        x = self.qgru(x)
        x = self.fc(x)
        # セグメントを時間軸方向に連結してから畳み込むことで、境界をまたぐ受容野を確保し波形を連続にする
        x = x.reshape(batch, length, self.initial_channel, -1)
        x = x.transpose(1, 2).reshape(batch, self.initial_channel, -1)
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)
        x = self.act(x)
        x = self.conv_final(x)
        x = self.act_final(x)
        x = x.reshape(batch, length, -1)
        return x

class VPMAutoEncoder(nn.Module):
    def __init__(self, waveform_length: int, dim_content: int, dim_print: int, dim: int, dim_hidden: int, num_layers: int):
        super().__init__()
        self.content_encoder = Encoder(waveform_length=waveform_length, dim=dim, dim_out=dim_content)
        self.print_encoder = Encoder(waveform_length=waveform_length, dim=dim, dim_out=dim_print)
        self.decoder = Decoder(waveform_length=waveform_length, dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content = self.content_encoder(x)
        voice_print = self.print_encoder(x)
        x = torch.cat((content, voice_print), dim=-1)
        x = self.decoder(x)
        return x, content, voice_print

    def upside_down(self, content: torch.Tensor, voice_print: torch.Tensor) -> torch.Tensor:
        x = torch.cat((content, voice_print), dim=-1)
        x = self.decoder(x)
        content_reconstructed = self.content_encoder(x)
        voice_print_reconstructed = self.print_encoder(x)
        return content_reconstructed, voice_print_reconstructed

class AutoEncoder(nn.Module):
    def __init__(self, decoder_type: str = 'hifigan', waveform_length: int = 2048, num_latent_frames: int = 4):
        super().__init__()
        assert waveform_length % num_latent_frames == 0, "waveform_length must be divisible by num_latent_frames"
        self.waveform_length = waveform_length
        self.encoder = Encoder(waveform_length=waveform_length, num_latent_frames=num_latent_frames)
        # デコーダは潜在フレーム1本あたり waveform_length / num_latent_frames サンプルを生成する
        frame_length = waveform_length // num_latent_frames
        if decoder_type == 'ddsp':
            self.decoder = Decoder(waveform_length=frame_length)
        elif decoder_type == 'hifigan':
            self.decoder = HiFiGANDecoder(waveform_length=frame_length)
        else:
            raise ValueError(f"unknown decoder_type: {decoder_type}")

    def forward(self, x):
        batch, length, _ = x.shape
        latent = self.encoder(x)
        x = self.decoder(latent)
        x = x.reshape(batch, length, self.waveform_length)
        return x, latent