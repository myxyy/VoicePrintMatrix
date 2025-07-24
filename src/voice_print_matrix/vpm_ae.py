from voice_print_matrix.qgru import QGRUModel
import torch.nn as nn
import torch
import torchaudio

class Encoder(nn.Module):
    def __init__(self, waveform_length=2048, dim=1024, dim_out=1024):
        super().__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=512, hop_length=256, n_mels=128, center=False)
        num_steps = (waveform_length - 512) // 256 + 1
        self.qgru = QGRUModel(dim_in=128 * num_steps, dim_out=dim_out, dim=dim, dim_hidden=2048, num_layers=4)

    def forward(self, x):
        batch, length, _ = x.shape
        mel_spectrogram = self.transform(x)
        x = mel_spectrogram.reshape(batch, length, -1)
        x = self.qgru(x)
        return x

class Decoder(nn.Module):
    def __init__(self, waveform_length=2048, dim=1024, num_oscillators=32):
        super().__init__()
        self.qgru = QGRUModel(dim_in=1024, dim_out=1024, dim=dim, dim_hidden=2048, num_layers=4)
        self.z_arg = nn.Parameter(torch.randn(num_oscillators))
        self.fs_fc = nn.Parameter(torch.randn(num_oscillators, dim, waveform_length) * dim ** -0.5)
        self.amp_fc = nn.Parameter(torch.randn(num_oscillators, dim, waveform_length) * dim ** -0.5)
        #nn.init.zeros_(self.fc_log_amp_init.weight)
        #nn.init.zeros_(self.fc_log_amp_init.bias)
        self.waveform_length = waveform_length

    def forward(self, x):
        batch, length, dim = x.shape
        x = self.qgru(x)
        x = x.reshape(batch * length, dim)
        fs = torch.einsum("bd, odw -> bow", x, self.fs_fc)
        fs = torch.cumsum(fs, dim=-1)
        amp = torch.einsum("bd, odw -> bow", x, self.amp_fc)
        arg = fs + self.z_arg[None,:,None]
        base_wave = torch.exp(1j * arg * torch.pi)  # Complex exponential
        waveform = amp * base_wave
        return waveform.mean(dim=1).real.reshape(batch, length, self.waveform_length)

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
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent