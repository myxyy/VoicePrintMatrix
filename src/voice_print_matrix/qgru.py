import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FFNSwiGLU(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim_ff_hidden)
        self.fc_act = nn.Linear(dim, dim_ff_hidden)
        self.fc_out = nn.Linear(dim_ff_hidden, dim)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x) * self.act(self.fc_act(x))
        x = self.fc_out(x)
        return x


def scan(a: Tensor, b: Tensor) -> Tensor:
    _, length = a.shape
    if length == 1:
        return b
    is_odd = length % 2 == 1
    a_even = a[:, : -1 if is_odd else None : 2]
    a_odd = a[:, 1::2]
    b_even = b[:, : -1 if is_odd else None : 2]
    b_odd = b[:, 1::2]
    mask_odd = torch.zeros(length, device=a.device, dtype=a.dtype)
    mask_odd[1::2] = 1
    mask_odd = mask_odd[None, :]
    b_new = torch.addcmul(
        torch.addcmul(b, b, mask_odd, value=-1),
        F.pad(
            scan(a_odd * a_even, torch.addcmul(b_odd, a_odd, b_even)).repeat_interleave(
                2, dim=1
            ),
            (0, 1) if is_odd else (0, 0),
            value=0,
        ),
        mask_odd,
    )
    b_odd_new = b_new[:, 1 : None if is_odd else -1 : 2]
    a_even_new = a[:, 2::2]
    mask_even = torch.zeros(length, device=a.device, dtype=a.dtype)
    mask_even[2::2] = 1
    mask_even = mask_even[None, :]
    b_new = torch.addcmul(
        b_new,
        F.pad(
            (a_even_new * b_odd_new).repeat_interleave(2, dim=1),
            (1, 0) if is_odd else (1, 1),
            value=0,
        ),
        mask_even,
    )
    return b_new


class QGRULayer(nn.Module):
    def __init__(self, dim: int, dim_hidden: int):
        super().__init__()
        self.dim = dim
        self.dim_hidden = dim_hidden
        self.fc_forget = nn.Linear(dim, dim_hidden)
        self.fc_input = nn.Linear(dim, dim_hidden)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc_out = nn.Linear(dim_hidden, dim)

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch, len, dim = x.shape

        remember = F.sigmoid(self.fc_forget(x)) * torch.linspace(0.0, 1.0, self.dim_hidden, device=x.device)[None, None, :]
        forget = 1 - remember

        input = self.tanh(self.fc_input(x))
        h_inner_chunk = (
            scan(
                forget.transpose(2, 1).reshape(batch * self.dim_hidden, len),
                (input * remember).transpose(2, 1).reshape(batch * self.dim_hidden, len),
            )
            .reshape(batch, self.dim_hidden, len)
            .transpose(2, 1)
        )

        h = torch.addcmul(h_inner_chunk, hidden[:, None, :], forget.cumprod(1))
        y = self.fc_out(h)

        return y, h[:, -1, :]


class QGRUBlock(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, dropout: float):
        super().__init__()
        self.qlstm = QGRULayer(dim, dim_hidden)
        self.ffn = FFNSwiGLU(dim, dim_hidden)
        self.norm_qlstm = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        x_ = x
        x = self.norm_qlstm(x)
        x, hidden = self.qlstm(x, hidden)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x, hidden

class QGRUModel(nn.Module):
    def __init__(
        self,
        dim_in: int = 1024,
        dim_out: int = 1024,
        dim: int = 1024,
        dim_hidden: int = 2048,
        num_layers: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [QGRUBlock(dim, dim_hidden, dropout) for _ in range(num_layers)]
        )
        self._hidden_init = nn.Parameter(
            torch.zeros(num_layers, dim_hidden)
        )
        self.fc_in = nn.Linear(dim_in, dim)
        self.fc_out_norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, dim_out)

    def hidden_init(self, batch):
        return self._hidden_init[None, :].expand(batch, -1, -1)

    def forward_with_hidden(self, x: Tensor, hidden: Tensor) -> Tensor:
        hidden_next = []
        x = self.fc_in(x)
        for i, layer in enumerate(self.layers):
            x, hidden_next_layer = layer(x, hidden[:, i])
            hidden_next.append(hidden_next_layer)
        hidden_next = torch.stack(hidden_next, dim=1)
        x = self.fc_out_norm(x)
        x = self.fc_out(x)
        return x, hidden_next

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        hidden = self.hidden_init(batch)

        x, _ = self.forward_with_hidden(x, hidden)
        return x
