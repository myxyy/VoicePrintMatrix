from voice_print_matrix.qgru import QGRUModel
import torch.nn as nn
import torch
import torchaudio

class Encoder(nn.Module):
    def __init__(self, waveform_length=2048, dim=1024, dim_hidden=2048, num_layers=4, dim_out=1024, n_mels=64):
        super().__init__()
        sample_rate = 22050
        n_fft = 512
        hop_length = 256
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, f_max=sample_rate // 2, hop_length=hop_length, n_mels=n_mels, center=False)
        num_steps = (waveform_length - n_fft) // hop_length + 1
        self.qgru = QGRUModel(dim_in=n_mels * num_steps, dim_out=dim_out, dim=dim, dim_hidden=dim_hidden, num_layers=num_layers)

    def forward(self, x):
        batch, length, _ = x.shape
        mel_spectrogram = self.transform(x)
        x = mel_spectrogram.reshape(batch, length, -1)
        x = self.qgru(x)
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
    def __init__(self, waveform_length=2048, dim=1024, dim_hidden=2048, num_layers=4, num_oscillators=16, sample_rate=22050):
        super().__init__()
        self.num_oscillators = num_oscillators
        self.waveform_length = waveform_length
        self.sample_rate = sample_rate
        self.qgru = QGRUModel(dim_in=dim, dim_out=dim, dim=dim, dim_hidden=dim_hidden, num_layers=num_layers)
        self.z_arg = nn.Linear(dim, num_oscillators)
        self.fs_fc = nn.Linear(dim, waveform_length)
        self.log_fs_scale = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.fs_fc.weight)
        nn.init.zeros_(self.fs_fc.bias)
        #self.fs_fc = MLP(dim, waveform_length, dim_hidden=dim_hidden)
        self.amp_fc = nn.Parameter(torch.randn(num_oscillators, dim, waveform_length) * dim ** -0.5)
        self.amp_whole_fc = nn.Linear(dim, waveform_length)
        #self.norm = nn.LayerNorm(dim)
        #self.fc_noise_1 = nn.Linear(dim + waveform_length, dim_hidden + waveform_length)
        #self.fc_noise_2 = nn.Linear(dim_hidden + waveform_length, waveform_length)
        self.amp_filter = nn.Parameter(torch.randn(dim, num_oscillators, waveform_length) * dim ** -0.5)
        self.amp_whole_filter = nn.Linear(dim, waveform_length)
        self.fs_filter = nn.Linear(dim, waveform_length)
        self.act = nn.SiLU()
        #nn.init.zeros_(self.fc_log_amp_init.weight)
        #nn.init.zeros_(self.fc_log_amp_init.bias)

    def forward(self, x):
        batch, length, dim = x.shape
        x = self.qgru(x)
        x = x.reshape(batch * length, dim)

        fs = self.fs_fc(x)

        fs_fft = torch.fft.rfft(nn.functional.pad(fs, (0, self.waveform_length), "constant", 0), dim=-1)
        fs_filter = torch.softmax(self.fs_filter(x), dim=-1)
        fs_filter_fft = torch.fft.rfft(nn.functional.pad(fs_filter, (0, self.waveform_length), "constant", 0), dim=-1)
        fs = torch.fft.irfft(fs_fft * fs_filter_fft, n=self.waveform_length, dim=-1)

        fs = torch.sigmoid(fs) * torch.exp(self.log_fs_scale)
        fs = torch.cumsum(fs, dim=-1)
        fs = fs[:,None,:] * (torch.arange(self.num_oscillators, device=x.device) + 1)[None,:,None]

        amp = torch.einsum("bd, odw -> bow", x, self.amp_fc)
        #amp = -self.act(amp)
        #amp[:,0,:] = 0  # Set the first oscillator's amplitude to zero
        amp = torch.softmax(amp, dim=1)

        amp_fft = torch.fft.rfft(nn.functional.pad(amp, (0, self.waveform_length), "constant", 0), dim=-1)
        amp_filter = torch.softmax(torch.einsum("bd, dow -> bow", x, self.amp_filter), dim=-1)
        amp_filter_fft = torch.fft.rfft(nn.functional.pad(amp_filter, (0, self.waveform_length), "constant", 0), dim=-1)
        amp = torch.fft.irfft(amp_fft * amp_filter_fft, n=self.waveform_length, dim=-1)

        amp_whole = self.amp_whole_fc(x)

        amp_whole_fft = torch.fft.rfft(nn.functional.pad(amp_whole, (0, self.waveform_length), "constant", 0), dim=-1)
        amp_whole_filter = torch.softmax(self.amp_whole_filter(x), dim=-1)
        amp_whole_filter_fft = torch.fft.rfft(nn.functional.pad(amp_whole_filter, (0, self.waveform_length), "constant", 0), dim=-1)
        amp_whole = torch.fft.irfft(amp_whole_fft * amp_whole_filter_fft, n=self.waveform_length, dim=-1)

        z_arg = self.z_arg(x)
        arg = fs + z_arg[:,:,None]
        base_wave = torch.sin(arg * torch.pi)  # Complex exponential
        waveform = amp * base_wave
        waveform = amp_whole * waveform.sum(dim=1)
        #noise = self.fc_noise_2(self.act(self.fc_noise_1(torch.cat((x, waveform), dim=-1))))
        #return (waveform + noise).reshape(batch, length, self.waveform_length)
        return waveform.reshape(batch, length, self.waveform_length)

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