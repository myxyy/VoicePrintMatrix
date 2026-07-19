import torch
import torch.nn as nn
from torch.fft import fft


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss (spectral convergence + log-magnitude L1).

    Hann窓・75%オーバーラップのSTFTを複数のFFTサイズで計算し、
    振幅スペクトルの spectral convergence と log振幅のL1を合計する。
    入力波形は (batch, time)。時間軸に連結した波形を渡せば、
    窓がセグメント境界をまたぐため境界の不連続も損失に反映される。
    """
    def __init__(self, n_ffts=(64, 128, 256, 512, 1024, 2048), eps=1e-5):
        super().__init__()
        self.n_ffts = n_ffts
        self.eps = eps
        for n_fft in n_ffts:
            self.register_buffer(f'window_{n_fft}', torch.hann_window(n_fft), persistent=False)

    def _magnitude(self, waveform, n_fft):
        window = getattr(self, f'window_{n_fft}')
        spec = torch.stft(waveform, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                          window=window, center=True, return_complex=True)
        return spec.abs()

    def forward(self, pred, target):
        loss = pred.new_zeros(())
        for n_fft in self.n_ffts:
            mag_pred = self._magnitude(pred, n_fft)
            mag_target = self._magnitude(target, n_fft)
            loss_sc = torch.norm(mag_target - mag_pred, p='fro') / torch.norm(mag_target, p='fro').clamp(min=self.eps)
            loss_mag = nn.functional.l1_loss(torch.log(mag_pred.clamp(min=self.eps)), torch.log(mag_target.clamp(min=self.eps)))
            loss = loss + loss_sc + loss_mag
        return loss / len(self.n_ffts)


def multiscale_spectrum(waveform, min_length=1):
    """
    Compute the multiscale spectrum of a waveform.
    
    Args:
        waveform (torch.Tensor): Input waveform tensor of shape (batch, length) where length is power of 2.
        
    Returns:
        torch.Tensor: Multiscale spectrum tensor of shape (batch_size, num_scales, length) where num_scales is log2(length).
    """
    batch, length = waveform.shape
    assert (length & (length - 1)) == 0, "Length must be a power of 2"
    waveform_fft = fft(waveform, dim=-1).abs()
    if length <= min_length:
        return waveform_fft.unsqueeze(1)  # Return as (batch, 1, length)
    else:
        waveform_half = waveform.reshape(batch * 2, length // 2)
        remaining_spectrum = multiscale_spectrum(waveform_half, min_length=min_length)
        remaining_spectrum = remaining_spectrum.reshape(batch, 2, -1, length // 2).transpose(1, 2).reshape(batch, -1, length)
        return torch.cat((waveform_fft.unsqueeze(1), remaining_spectrum), dim=1)


