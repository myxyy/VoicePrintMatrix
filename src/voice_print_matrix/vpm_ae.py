from voice_print_matrix.qgru import QGRUModel
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, waveform_length=2048, dim=1024):
        super().__init__()
        self.fc = nn.Linear(waveform_length * 2, dim)

    def forward(self, x):
        fft = torch.fft.fft(x, dim=-1)
        return self.fc(torch.cat((fft.real, fft.imag), dim=-1))

class Decoder(nn.Module):
    def __init__(self, waveform_length=2048, dim=1024, num_oscillators=128):
        super().__init__()
        self.z_arg = nn.Parameter(torch.randn(num_oscillators))
        self.z_init_arg_fc = nn.Linear(dim, num_oscillators)
        self.z_init_amp_fc = nn.Linear(dim, num_oscillators)
        #nn.init.zeros_(self.fc_log_amp_init.weight)
        #nn.init.zeros_(self.fc_log_amp_init.bias)
        self.waveform_length = waveform_length

    def forward(self, x):
        batch, length, dim = x.shape
        x = x.reshape(batch * length, dim)
        z_init_arg = self.z_init_arg_fc(x) * torch.pi
        z_init_amp = self.z_init_amp_fc(x)
        z_init = z_init_amp * torch.exp(1j * z_init_arg)
        z = torch.exp(1j * self.z_arg * torch.pi)  # Complex exponential
        waveform = z_init[:,:,None] * pow(z[:,None], torch.arange(self.waveform_length, device=x.device)[None,:])[None, : ,:]
        return waveform.mean(dim=1).real.reshape(batch, length, self.waveform_length)

class VPMAutoEncoder(nn.Module):
    def __init__(self, waveform_length: int, dim_content: int, dim_print: int, dim: int, dim_hidden: int, num_layers: int):
        super().__init__()
        self.content_encoder = nn.Sequential(
            Encoder(waveform_length=waveform_length, dim=dim),
            QGRUModel(
                dim_in=dim,
                dim_out=dim_content,
                dim=dim,
                dim_hidden=dim_hidden,
                num_layers=num_layers
            )
        )
        self.print_encoder = nn.Sequential(
            Encoder(waveform_length=waveform_length, dim=dim),
            QGRUModel(
                dim_in=dim,
                dim_out=dim_print,
                dim=dim,
                dim_hidden=dim_hidden,
                num_layers=num_layers
            )
        )
        self.decoder = nn.Sequential(
            QGRUModel(
                dim_in=dim_content + dim_print,
                dim_out=dim,
                dim=dim,
                dim_hidden=dim_hidden,
                num_layers=num_layers
            ),
            Decoder(waveform_length=waveform_length, dim=dim)
        )

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
