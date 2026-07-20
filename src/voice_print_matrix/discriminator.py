import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        channels = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.convs.append(weight_norm(nn.Conv2d(channels[i], channels[i + 1], (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0))))
        self.convs.append(weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), (1, 1), padding=((kernel_size - 1) // 2, 0))))
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), (1, 1), padding=(1, 0)))
        self.act = nn.SiLU()

    def forward(self, x):
        batch, _, length = x.shape
        if length % self.period != 0:
            pad = self.period - (length % self.period)
            x = F.pad(x, (0, pad), "reflect")
            length = length + pad
        x = x.reshape(batch, 1, length // self.period, self.period)
        features = []
        for conv in self.convs:
            x = self.act(conv(x))
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1, -1), features


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.act = nn.SiLU()

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = self.act(conv(x))
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1, -1), features


class HiFiGANDiscriminator(nn.Module):
    """
    HiFi-GANのMulti-Period + Multi-Scale Discriminator。
    入力は (batch, 1, time) の波形クロップ。
    realとfakeは連結して1回のforwardで処理する(DDPは1回のbackwardにつき
    1回のforwardしか扱えないため、real/fakeを別々にforwardすると壊れる)。
    """
    def __init__(self, periods=[2, 3, 5, 7, 11], num_scales=3):
        super().__init__()
        self.period_discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])
        self.scale_discriminators = nn.ModuleList([ScaleDiscriminator() for _ in range(num_scales)])
        self.pool = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, real, fake):
        num_real = real.shape[0]
        x = torch.cat((real, fake), dim=0)
        logits_real, logits_fake, features_real, features_fake = [], [], [], []

        def append(logit, features):
            logits_real.append(logit[:num_real])
            logits_fake.append(logit[num_real:])
            features_real.append([f[:num_real] for f in features])
            features_fake.append([f[num_real:] for f in features])

        for discriminator in self.period_discriminators:
            append(*discriminator(x))
        y = x
        for i, discriminator in enumerate(self.scale_discriminators):
            if i > 0:
                y = self.pool(y)
            append(*discriminator(y))
        return logits_real, logits_fake, features_real, features_fake


def discriminator_loss(logits_real, logits_fake):
    # LSGAN形式
    loss = 0.0
    for logit_real, logit_fake in zip(logits_real, logits_fake):
        loss = loss + torch.mean((logit_real - 1.0) ** 2) + torch.mean(logit_fake ** 2)
    return loss


def generator_adversarial_loss(logits_fake):
    loss = 0.0
    for logit_fake in logits_fake:
        loss = loss + torch.mean((logit_fake - 1.0) ** 2)
    return loss


def feature_matching_loss(features_real, features_fake):
    loss = 0.0
    for feats_real, feats_fake in zip(features_real, features_fake):
        for feat_real, feat_fake in zip(feats_real, feats_fake):
            loss = loss + F.l1_loss(feat_real.detach(), feat_fake)
    return loss
