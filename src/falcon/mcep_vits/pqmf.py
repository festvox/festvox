"""Pseudo Quadrature Mirror Filter (PQMF) for subband analysis/synthesis.

Used by MB-iSTFT-VITS multi-band decoder paths. Not used by the MCEP decoder,
but kept for compatibility with the shared SynthesizerTrn and training loop.

Reference: "Near-perfect-reconstruction pseudo-QMF banks" (IEEE, 1994)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal.windows import kaiser


def _design_prototype_filter(taps=62, cutoff_ratio=0.15, beta=9.0):
    """Design Kaiser-windowed lowpass prototype filter for PQMF.

    Args:
        taps: filter length (must be even).
        cutoff_ratio: cutoff frequency as fraction of pi.
        beta: Kaiser window shape parameter.
    Returns:
        prototype filter coefficients, shape [taps+1].
    """
    omega_c = np.pi * cutoff_ratio
    n = np.arange(taps + 1)
    mid = taps / 2.0
    with np.errstate(invalid='ignore'):
        h = np.sin(omega_c * (n - mid)) / (np.pi * (n - mid))
    h[taps // 2] = cutoff_ratio  # L'Hopital at center tap
    return h * kaiser(taps + 1, beta)


class PQMF(nn.Module):
    """4-subband cosine-modulated PQMF filterbank."""

    def __init__(self, device, subbands=4, taps=62, cutoff_ratio=0.15,
                 beta=9.0):
        super().__init__()
        self.subbands = subbands

        proto = _design_prototype_filter(taps, cutoff_ratio, beta)
        length = len(proto)

        # Cosine-modulated analysis/synthesis filter pairs
        h_a = np.zeros((subbands, length))
        h_s = np.zeros((subbands, length))
        for k in range(subbands):
            for n in range(length):
                angle = (2 * k + 1) * np.pi / (2 * subbands) * \
                        (n - (taps - 1) / 2.0)
                sign = (-1) ** k * np.pi / 4
                h_a[k, n] = 2 * proto[n] * np.cos(angle + sign)
                h_s[k, n] = 2 * proto[n] * np.cos(angle - sign)

        # Analysis: [subbands, 1, length], Synthesis: [1, subbands, length]
        a_filt = torch.from_numpy(h_a).float().unsqueeze(1).to(device)
        s_filt = torch.from_numpy(h_s).float().unsqueeze(0).to(device)

        self.register_buffer('analysis_filter', a_filt)
        self.register_buffer('synthesis_filter', s_filt)

        # Polyphase downsampling/upsampling via identity conv
        updown = torch.zeros(subbands, subbands, subbands).float().to(device)
        for k in range(subbands):
            updown[k, k, 0] = 1.0
        self.register_buffer('updown_filter', updown)

        self.pad_fn = nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """Decompose fullband signal into subbands.

        Args:
            x: [B, 1, T] fullband signal.
        Returns:
            [B, subbands, T // subbands] subband signals.
        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Reconstruct fullband signal from subbands.

        Args:
            x: [B, subbands, T_sub] subband signals.
        Returns:
            [B, 1, T] reconstructed fullband signal.
        """
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands,
                                stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)
