"""SNAC codec wrapper — encode/decode audio, flatten/unflatten multi-scale tokens."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio


class SNACCodec:
    """Wrapper around hubertsiuzdak/snac_24khz.

    SNAC produces 3 levels of codes at 12/24/48 Hz.
    We flatten them into a single interleaved sequence:
        [c0, m0, m1, f0, f1, f2, f3, c1, m2, m3, f4, f5, f6, f7, ...]
    giving 7 tokens per coarse frame = 84 tokens/sec.
    """

    SAMPLE_RATE = 24000
    COARSE_HZ = 12
    N_LEVELS = 3
    CODEBOOK_SIZE = 4096
    TOKENS_PER_FRAME = 7  # 1 coarse + 2 mid + 4 fine

    def __init__(self, device: str = "cuda"):
        from snac import SNAC
        self.device = device
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> List[Tuple[int, int]]:
        """Encode waveform to flat list of (level, code) pairs.

        Args:
            waveform: [1, 1, T] tensor at 24kHz

        Returns:
            List of (level, code) tuples in interleaved order
        """
        waveform = waveform.to(self.device)
        codes = self.model.encode(waveform)
        # codes: [level0 [1, N], level1 [1, 2N], level2 [1, 4N]]
        return self._flatten_codes(codes)

    def encode_file(self, wav_path: str, target_sr: int = 24000) -> List[Tuple[int, int]]:
        """Load a wav file and encode to flat (level, code) pairs."""
        waveform, sr = torchaudio.load(wav_path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = waveform.unsqueeze(0)  # [1, 1, T]
        return self.encode(waveform)

    @torch.no_grad()
    def decode(self, flat_codes: List[Tuple[int, int]]) -> torch.Tensor:
        """Decode flat (level, code) pairs back to waveform.

        Returns:
            [1, 1, T] waveform tensor at 24kHz
        """
        codes = self._unflatten_codes(flat_codes)
        audio = self.model.decode(codes)
        return audio

    def decode_to_file(self, flat_codes: List[Tuple[int, int]], wav_path: str):
        """Decode flat codes and save to wav file."""
        audio = self.decode(flat_codes)
        torchaudio.save(wav_path, audio.squeeze(0).cpu(), self.SAMPLE_RATE)

    def _flatten_codes(self, codes: list[torch.Tensor]) -> List[Tuple[int, int]]:
        """Flatten multi-scale SNAC codes into interleaved (level, code) pairs.

        Pattern per coarse frame: [c, m, m, f, f, f, f] = 7 tokens
        """
        c = codes[0][0].cpu().tolist()  # [N]
        m = codes[1][0].cpu().tolist()  # [2N]
        f = codes[2][0].cpu().tolist()  # [4N]

        flat = []
        n_coarse = len(c)
        for i in range(n_coarse):
            flat.append((0, c[i]))                # 1 coarse
            flat.append((1, m[2 * i]))             # 2 mid
            flat.append((1, m[2 * i + 1]))
            flat.append((2, f[4 * i]))             # 4 fine
            flat.append((2, f[4 * i + 1]))
            flat.append((2, f[4 * i + 2]))
            flat.append((2, f[4 * i + 3]))
        return flat

    def _unflatten_codes(self, flat: List[Tuple[int, int]]) -> list[torch.Tensor]:
        """Unflatten interleaved (level, code) pairs back to SNAC format."""
        c, m, f = [], [], []
        for level, code in flat:
            if level == 0:
                c.append(code)
            elif level == 1:
                m.append(code)
            elif level == 2:
                f.append(code)

        device = self.device
        return [
            torch.tensor([c], dtype=torch.long, device=device),
            torch.tensor([m], dtype=torch.long, device=device),
            torch.tensor([f], dtype=torch.long, device=device),
        ]

    def save_codes(self, flat_codes: List[Tuple[int, int]], path: str | Path):
        """Save flat codes as numpy array [N, 2] for efficient storage."""
        arr = np.array(flat_codes, dtype=np.int16)
        np.save(str(path), arr)

    @staticmethod
    def load_codes(path: str | Path) -> List[Tuple[int, int]]:
        """Load flat codes from numpy file."""
        arr = np.load(str(path))
        return [(int(row[0]), int(row[1])) for row in arr]
