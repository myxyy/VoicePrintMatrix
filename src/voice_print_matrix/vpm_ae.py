from voice_print_matrix.qgru import QGRUModel
import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.fft.fft(x, dim=-1).abs()

class Decoder(nn.Module):
    def __init__(self, length=2048):
        super().__init__()
        self.fc_real = nn.Linear(length, length)
        self.fc_imag = nn.Linear(length, length)

    def forward(self, x):
        real = self.fc_real(x)
        imag = self.fc_imag(x)
        complex_tensor = torch.complex(real, imag)
        return torch.fft.ifft(complex_tensor, dim=-1).real

class VPMAutoEncoder(nn.Module):
    def __init__(self, dim_token: int, dim_content: int, dim_print: int, dim: int, dim_hidden: int, num_layers: int):
        super().__init__()
        self.content_encoder = nn.Sequential(
            Encoder(),
            QGRUModel(
                dim_in=dim_token,
                dim_out=dim_content,
                dim=dim,
                dim_hidden=dim_hidden,
                num_layers=num_layers
            )
        )
        self.print_encoder = nn.Sequential(
            Encoder(),
            QGRUModel(
                dim_in=dim_token,
                dim_out=dim_print,
                dim=dim,
                dim_hidden=dim_hidden,
                num_layers=num_layers
            )
        )
        self.decoder = nn.Sequential(
            QGRUModel(
                dim_in=dim_content + dim_print,
                dim_out=dim_token,
                dim=dim,
                dim_hidden=dim_hidden,
                num_layers=num_layers
            ),
            Decoder(dim_token)
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
