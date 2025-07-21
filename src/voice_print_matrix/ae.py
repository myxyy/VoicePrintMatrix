import torch
import torch.nn as nn

#class ResConvStack(nn.Module):
#    def __init__(self, channel, depth):
#        super().__init__()
#        self.conv_list = nn.ModuleList([nn.Conv1d(channel, channel, 5, stride=1, padding=2) for _ in range(depth)])
#        self.act = nn.ReLU()
#
#    def forward(self, x):
#        for conv in self.conv_list:
#            x_ = x
#            x = conv(x)
#            x = self.act(x)
#            x = x + x_
#        return x
#
#class ResConvTransStack(nn.Module):
#    def __init__(self, channel, depth):
#        super().__init__()
#        self.conv_list = nn.ModuleList([nn.Sequential(
#            nn.ConvTranspose1d(channel, channel, 5, stride=1, padding=2),
#            nn.SiLU(),
#            nn.Conv1d(channel, channel, 5, stride=1, padding=2)
#        ) for _ in range(depth)])
#
#    def forward(self, x):
#        for conv in self.conv_list:
#            x_ = x
#            x = conv(x)
#            x = x + x_
#        return x
#
#
#class Encoder(nn.Module):
#    def __init__(self):
#        super().__init__()
#        channel = 32
#        self.conv1 = nn.Conv1d(1, channel, 4, stride=2, padding=1)
#        self.conv2 = nn.Conv1d(channel, channel, 4, stride=2, padding=1)
#        self.conv3 = nn.Conv1d(channel, channel, 4, stride=2, padding=1)
#        self.conv4 = nn.Conv1d(channel, channel, 4, stride=2, padding=1)
#        self.res_stack = ResConvStack(channel, 4) 
#        self.conv_out = nn.Conv1d(channel, 4, 5, stride=1, padding=2)
#        self.act = nn.SiLU()
#
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.act(x)
#        x = self.conv2(x)
#        x = self.act(x)
#        x = self.conv3(x)
#        x = self.act(x)
#        x = self.conv4(x)
#        x = self.act(x)
#        x = self.res_stack(x)
#        x = self.act(x)
#        x = self.conv_out(x)
#        return x
#
#class Decoder(nn.Module):
#    def __init__(self):
#        super().__init__()
#        channel = 32
#        self.conv_in = nn.ConvTranspose1d(4, channel, 5, stride=1, padding=2)
#        self.res_stack = ResConvTransStack(channel, 4) 
#        self.conv4 = nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1)
#        self.conv3 = nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1)
#        self.conv2 = nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1)
#        self.conv1 = nn.ConvTranspose1d(channel, 1, 4, stride=2, padding=1)
#        self.act = nn.SiLU()
#
#    def forward(self, x):
#        x = self.conv_in(x)
#        x = self.act(x)
#        x = self.res_stack(x)
#        x = self.act(x)
#        x = self.conv4(x)
#        x = self.act(x)
#        x = self.conv3(x)
#        x = self.act(x)
#        x = self.conv2(x)
#        x = self.act(x)
#        x = self.conv1(x)
#        return x
#
#class AutoEncoder(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.encoder = Encoder()
#        self.decoder = Decoder()
#
#    def forward(self, x):
#        latent = self.encoder(x)
#        x = self.decoder(latent)
#        return x, latent

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

class AutoEncoder(nn.Module):
    def __init__(self, dim_segment=2048, dim_hidden=8192, dim_token=512):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim_segment, dim_hidden), nn.SiLU(), nn.Linear(dim_hidden, dim_token), nn.LayerNorm(dim_token))
        self.decoder = nn.Sequential(nn.Linear(dim_token, dim_hidden), nn.SiLU(), nn.Linear(dim_hidden, dim_segment))

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent