"""STFT wrapper using torch.stft/torch.istft."""

import torch
import torch.nn as nn


class TorchSTFT(nn.Module):
    """Thin wrapper around torch.stft/istft for magnitude-phase decomposition."""

    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer(
            'window',
            getattr(torch, f'{window}_window')(win_length).float()
        )

    def transform(self, input_data):
        """Compute magnitude and phase from waveform.

        Args:
            input_data: [B, T] waveform
        Returns:
            magnitude: [B, n_freq, n_frames]
            phase: [B, n_freq, n_frames]
        """
        spec = torch.stft(
            input_data, self.filter_length, self.hop_length, self.win_length,
            window=self.window, return_complex=True,
        )
        return torch.abs(spec), torch.angle(spec)

    def inverse(self, magnitude, phase):
        """Reconstruct waveform from magnitude and phase.

        Args:
            magnitude: [B, n_freq, n_frames]
            phase: [B, n_freq, n_frames]
        Returns:
            waveform: [B, 1, T]
        """
        spec = magnitude.float() * torch.exp(1j * phase.float())
        wav = torch.istft(
            spec, self.filter_length, self.hop_length, self.win_length,
            window=self.window.to(magnitude.device),
        )
        return wav.unsqueeze(-2)

    def forward(self, input_data):
        mag, phase = self.transform(input_data)
        return self.inverse(mag, phase)
