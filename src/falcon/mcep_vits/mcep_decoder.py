"""
MCEP-based decoder for VITS.

Replaces the conv upsampling + resblock pipeline with:
1. Deep ResBlock backbone for feature extraction
2. MCEP prediction head -> fixed mc2sp transform -> log magnitude
3. Magnitude refinement (bounded residual on MCEP log-mag)
4. Minimum phase computation + learned phase residual
5. Full-resolution iSTFT to generate waveform

No subbands, no PQMF, no conv upsampling.
Decoder params: ~3.9M (with D=192, 5 resblocks).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import pysptk


class MCEPBasis(nn.Module):
    """Pre-computes the mc2sp basis matrix for differentiable MCEP -> magnitude.

    The basis matrix B[n_freq, mcep_dim] satisfies:
        log_magnitude_spectrum = B @ mcep_coefficients

    This is exact because MCEPs are cepstral coefficients on a warped
    frequency axis, making the log-power spectrum linear in the MCEP
    coefficients.

    The matrix is computed once at init using pysptk.mc2sp on unit vectors
    and registered as a buffer (not a parameter).
    """

    def __init__(self, mcep_dim=40, n_fft=1024, alpha=0.455):
        super().__init__()
        self.mcep_dim = mcep_dim
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1
        self.alpha = alpha

        basis = self._compute_basis()
        self.register_buffer('basis_matrix', torch.from_numpy(basis))

    def _compute_basis(self):
        """Compute basis matrix using pysptk.mc2sp on unit vectors.

        For each unit vector e_i, mc2sp returns the power spectrum.
        Since log(power) = 2*log(magnitude), we store 0.5*log(power)
        as the basis for log-magnitude prediction.
        """
        basis = np.zeros((self.n_freq, self.mcep_dim), dtype=np.float32)
        for i in range(self.mcep_dim):
            mc = np.zeros(self.mcep_dim, dtype=np.float64)
            mc[i] = 1.0
            power_spec = pysptk.mc2sp(mc, alpha=self.alpha, fftlen=self.n_fft)
            log_mag = 0.5 * np.log(np.maximum(power_spec, 1e-10))
            basis[:, i] = log_mag.astype(np.float32)
        return basis

    def forward(self, mcep):
        """Convert MCEP coefficients to log magnitude spectrum.

        Args:
            mcep: [B, mcep_dim, T]
        Returns:
            log_mag: [B, n_freq, T]
        """
        # F.linear: input [B, T, mcep_dim] @ weight.T [mcep_dim, n_freq] -> [B, T, n_freq]
        return F.linear(mcep.transpose(1, 2), self.basis_matrix).transpose(1, 2)


def compute_min_phase(log_mag, n_fft):
    """Compute minimum phase from log magnitude spectrum via cepstral method.

    Given a log magnitude spectrum, the minimum phase is the unique phase
    that makes the signal causal and minimum-energy. It is computed by:
    1. IDFT of log magnitude -> real cepstrum
    2. Apply minimum-phase lifter (zero anti-causal part, double causal part)
    3. DFT of liftered cepstrum -> complex log spectrum
    4. Imaginary part = minimum phase

    This is exact, differentiable (FFT supports autograd), and adds 0 params.

    Args:
        log_mag: [B, n_freq, T] where n_freq = n_fft//2+1
        n_fft: FFT size (e.g. 1024)
    Returns:
        min_phase: [B, n_freq, T] minimum phase spectrum
    """
    # Put freq dim last for FFT: [B, n_freq, T] -> [B, T, n_freq]
    lm = log_mag.transpose(1, 2)

    # Real cepstrum via inverse real FFT
    # irfft treats input as one-sided DFT coefficients of a real signal
    cepstrum = torch.fft.irfft(lm, n=n_fft, dim=-1)  # [B, T, n_fft]

    # Minimum-phase lifter: keep DC and Nyquist, double causal, zero anti-causal
    lifter = torch.zeros(n_fft, device=cepstrum.device, dtype=cepstrum.dtype)
    lifter[0] = 1.0
    lifter[1:n_fft // 2] = 2.0
    lifter[n_fft // 2] = 1.0
    # indices n_fft//2+1 to n_fft-1 stay 0 (anti-causal zeroed out)

    cepstrum_min = cepstrum * lifter  # [B, T, n_fft]

    # DFT of min-phase cepstrum -> complex log spectrum
    log_spec_min = torch.fft.rfft(cepstrum_min, n=n_fft, dim=-1)  # [B, T, n_freq] complex

    # Imaginary part is the minimum phase
    min_phase = log_spec_min.imag  # [B, T, n_freq]

    # Transpose back: [B, T, n_freq] -> [B, n_freq, T]
    return min_phase.transpose(1, 2)


class MCEPDecoder(nn.Module):
    """MCEP-based decoder — deep ResBlock backbone for phase quality.

    Uses a deep ResBlock backbone (5x ResBlock1 by default) giving a large
    receptive field (>100 frames), with MCEP-based magnitude prediction,
    bounded magnitude refinement, and minimum-phase + learned phase residual.

    Architecture:
        z [B, C, T]
          -> project: Conv1d(C, D, k=1)
          -> 5x ResBlock1(D, k=3, dilation=(1,3,5))
          -> pad T -> T+1 (for iSTFT center=True)
          |
          +-- MCEP path:
          |     mcep_head(D, 40, k=1) -> MCEPBasis -> mcep_log_mag
          |
          +-- Magnitude refinement:
          |     mag_conv(D, D, k=3) + LReLU -> mag_head(D, 513, k=1)
          |     log_mag = mcep_log_mag + 0.3 * tanh(mag_refine)
          |     mag = exp(log_mag)
          |
          +-- Phase path:
                phase_conv1(D, D, k=3) + LReLU
                phase_conv2(D, D, k=3) + LReLU
                phase_head(D, 513, k=1)
                min_phase = compute_min_phase(log_mag)
                phase = min_phase + pi * sin(phase_residual)
          |
          iSTFT -> waveform

    Total decoder: ~3.9M (with D=192, 5 resblocks)
    """

    def __init__(self, initial_channel, mcep_dim=40, n_fft=1024,
                 hop_length=256, win_length=1024, alpha=0.455,
                 gin_channels=0, decoder_channels=192,
                 n_decoder_resblocks=5):
        super().__init__()
        from modules import ResBlock1

        self.initial_channel = initial_channel
        self.mcep_dim = mcep_dim
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1  # 513
        self.hop_length = hop_length
        self.win_length = win_length
        self.decoder_channels = decoder_channels

        # Speaker conditioning
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, initial_channel, 1)

        # Project to decoder channels
        self.project = weight_norm(
            nn.Conv1d(initial_channel, decoder_channels, kernel_size=1)
        )

        # Deep ResBlock backbone
        self.resblocks = nn.ModuleList()
        for _ in range(n_decoder_resblocks):
            self.resblocks.append(
                ResBlock1(decoder_channels, kernel_size=3, dilation=(1, 3, 5))
            )

        # MCEP prediction head
        self.mcep_head = nn.Conv1d(decoder_channels, mcep_dim, kernel_size=1)

        # Magnitude refinement (bounded residual on top of MCEP log-mag)
        self.mag_conv = weight_norm(
            nn.Conv1d(decoder_channels, decoder_channels, kernel_size=3, padding=1)
        )
        self.mag_head = nn.Conv1d(decoder_channels, self.n_freq, kernel_size=1)

        # Phase prediction (2-layer conv + head, residual on min-phase)
        self.phase_conv1 = weight_norm(
            nn.Conv1d(decoder_channels, decoder_channels, kernel_size=3, padding=1)
        )
        self.phase_conv2 = weight_norm(
            nn.Conv1d(decoder_channels, decoder_channels, kernel_size=3, padding=1)
        )
        self.phase_head = nn.Conv1d(decoder_channels, self.n_freq, kernel_size=1)

        # Fixed mc2sp basis matrix (not learnable)
        self.mcep_basis = MCEPBasis(mcep_dim, n_fft, alpha)

        # iSTFT window (registered as buffer for device tracking)
        window = torch.hann_window(win_length)
        self.register_buffer('window', window)

    def forward(self, x, g=None):
        """
        Args:
            x: [B, initial_channel, T] latent from posterior encoder
            g: [B, gin_channels, 1] speaker embedding (optional)
        Returns:
            y: [B, 1, T*hop_length] waveform
            None: placeholder for y_mb (no subbands)
        """
        T = x.shape[-1]

        # Speaker conditioning
        if g is not None:
            x = x + self.cond(g)

        # Project to decoder channels
        h = self.project(x)

        # Deep ResBlock backbone
        for resblock in self.resblocks:
            h = resblock(h)

        # Pad by 1 frame for iSTFT (T frames -> T+1 STFT frames needed
        # because center=True expects n_frames = signal_length/hop + 1)
        h = F.pad(h, (0, 1), mode='replicate')

        # MCEP prediction -> log magnitude (smooth spectral envelope)
        mcep = self.mcep_head(h)  # [B, mcep_dim, T+1]
        mcep_log_mag = self.mcep_basis(mcep)  # [B, n_freq, T+1]

        # Magnitude refinement (bounded residual for fine spectral detail)
        mag_h = self.mag_conv(h)
        mag_h = F.leaky_relu(mag_h, 0.1)
        mag_refine = self.mag_head(mag_h)  # [B, n_freq, T+1]
        log_mag = mcep_log_mag + 0.3 * torch.tanh(mag_refine)
        mag = torch.exp(log_mag)

        # Minimum phase from log magnitude (0 params, differentiable)
        min_phase = compute_min_phase(log_mag.float(), self.n_fft)  # [B, n_freq, T+1]

        # Phase residual prediction (2-layer conv)
        phase_h = self.phase_conv1(h)
        phase_h = F.leaky_relu(phase_h, 0.1)
        phase_h = self.phase_conv2(phase_h)
        phase_h = F.leaky_relu(phase_h, 0.1)
        phase_residual_raw = self.phase_head(phase_h)  # [B, n_freq, T+1]
        phase_residual = math.pi * torch.sin(phase_residual_raw)

        # Final phase = minimum phase + learned residual
        phase = min_phase + phase_residual

        # Combine into complex STFT and invert
        # Cast to float32 for numerical stability under fp16 autocast
        mag_f = mag.float()
        phase_f = phase.float()
        complex_spec = mag_f * torch.exp(1j * phase_f)

        y = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=T * self.hop_length
        )
        y = y.unsqueeze(1)  # [B, 1, T*hop_length]

        return y, None

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.project)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        remove_weight_norm(self.mag_conv)
        remove_weight_norm(self.phase_conv1)
        remove_weight_norm(self.phase_conv2)
