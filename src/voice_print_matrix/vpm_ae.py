from voice_print_matrix.qgru import QGRUModel
import torch.nn as nn
import torch

class VPMAutoEncoder(nn.Module):
    def __init__(self, dim_token: int, dim_content: int, dim_print: int, dim: int, dim_hidden: int, num_layers: int):
        super().__init__()
        self.content_encoder = QGRUModel(
            dim_in=dim_token,
            dim_out=dim_content,
            dim=dim,
            dim_hidden=dim_hidden,
            num_layers=num_layers
        )
        self.print_encoder = QGRUModel(
            dim_in=dim_token,
            dim_out=dim_print,
            dim=dim,
            dim_hidden=dim_hidden,
            num_layers=num_layers
        )
        self.decoder = QGRUModel(
            dim_in=dim_content + dim_print,
            dim_out=dim_token,
            dim=dim,
            dim_hidden=dim_hidden,
            num_layers=num_layers
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
