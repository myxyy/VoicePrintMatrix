import torch
from torch.fft import fft

def multiscale_spectrum(waveform, min_length=64):
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
        return waveform_fft.unsqueeze(1)  # Return as (batch, 1, length) if length is 1
    else:
        waveform_half = waveform.reshape(batch * 2, length // 2)
        remaining_spectrum = multiscale_spectrum(waveform_half)
        remaining_spectrum = remaining_spectrum.reshape(batch, 2, -1, length // 2).transpose(1, 2).reshape(batch, -1, length)
        return torch.cat((waveform_fft.unsqueeze(1), remaining_spectrum), dim=1)


