"""Multi-resolution STFT loss for GAN-based speech synthesis."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stft_mag(x, fft_size, hop_size, win_length, window):
    """Compute STFT magnitude spectrogram.

    Args:
        x: [B, T] input signal.
        fft_size: FFT size.
        hop_size: hop size.
        win_length: window length.
        window: window tensor.
    Returns:
        [B, n_frames, n_freq] magnitude spectrogram.
    """
    spec = torch.stft(
        x, fft_size, hop_size, win_length,
        window=window.to(x.device), return_complex=True,
    )
    # clamp to avoid sqrt(0) producing nan gradients
    return torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-7).transpose(2, 1)


class STFTLoss(nn.Module):
    """Single-resolution STFT loss (spectral convergence + log magnitude)."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600,
                 window="hann_window"):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer('window', getattr(torch, window)(win_length))

    def forward(self, x, y):
        """
        Args:
            x: [B, T] predicted signal.
            y: [B, T] ground truth signal.
        Returns:
            sc_loss: spectral convergence loss.
            mag_loss: log STFT magnitude L1 loss.
        """
        x_mag = _stft_mag(x, self.fft_size, self.shift_size, self.win_length,
                          self.window)
        y_mag = _stft_mag(y, self.fft_size, self.shift_size, self.win_length,
                          self.window)
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        mag_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss averaged over multiple (fft, hop, win) configs."""

    def __init__(self, fft_sizes=(1024, 2048, 512), hop_sizes=(120, 240, 50),
                 win_lengths=(600, 1200, 240), window="hann_window"):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(fs, hs, wl, window)
            for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, x, y):
        """
        Args:
            x: [B, T] predicted signal.
            y: [B, T] ground truth signal.
        Returns:
            sc_loss: mean spectral convergence loss across resolutions.
            mag_loss: mean log magnitude loss across resolutions.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for loss_fn in self.losses:
            sc_l, mag_l = loss_fn(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        n = len(self.losses)
        return sc_loss / n, mag_loss / n
