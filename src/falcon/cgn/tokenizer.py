"""CGN v2 unified tokenizer — linguistic + prosodic plan + SNAC audio tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


# ── Special tokens ──────────────────────────────────────────────────────────
PAD = 0
UNK = 1
BOS = 2
EOS = 3
PLAN_START = 4
AUDIO_START = 5
NUM_SPECIAL = 6

SPECIAL_TOKENS = {
    "<PAD>": PAD, "<UNK>": UNK, "<BOS>": BOS, "<EOS>": EOS,
    "<PLAN>": PLAN_START, "<AUDIO>": AUDIO_START,
}

# ── Prosodic plan token types ──────────────────────────────────────────────
# These define the "vocabulary" for the prosodic plan phase.
# Each token type has a prefix and a set of discrete values.

DURATION_BINS = [
    "dur:very_short", "dur:short", "dur:med_short", "dur:medium",
    "dur:med_long", "dur:long", "dur:very_long", "dur:extra_long",
]

PITCH_CONTOURS = [
    "pitch:flat", "pitch:rising", "pitch:falling", "pitch:rise_fall",
    "pitch:fall_rise", "pitch:high", "pitch:low", "pitch:question",
]

EMPHASIS_LEVELS = [
    "emph:unstressed", "emph:normal", "emph:stressed", "emph:contrastive",
]

PAUSE_TYPES = [
    "pause:none", "pause:short", "pause:medium", "pause:long",
]

STRUCTURAL_TOKENS = [
    "struct:word_start", "struct:phrase_start", "struct:phrase_end",
]

ALL_PROSODY_TOKENS = (
    DURATION_BINS + PITCH_CONTOURS + EMPHASIS_LEVELS
    + PAUSE_TYPES + STRUCTURAL_TOKENS
)


class CGNv2Tokenizer:
    """Unified tokenizer for CGN v2.

    Token layout:
        [0..5]                          Special tokens (PAD, UNK, BOS, EOS, PLAN, AUDIO)
        [6..6+N_ling-1]                 Linguistic tokens (phonemes + stress markers)
        [6+N_ling..6+N_ling+N_pros-1]   Prosodic plan tokens
        [6+N_ling+N_pros..]             SNAC audio tokens (3 levels * 4096)

    SNAC audio tokens are offset by level:
        Level 0 (coarse, 12Hz):  audio_offset + 0*4096 + code
        Level 1 (mid, 24Hz):     audio_offset + 1*4096 + code
        Level 2 (fine, 48Hz):    audio_offset + 2*4096 + code
    """

    def __init__(
        self,
        phoneme_list: List[str],
        stress_markers: Optional[List[str]] = None,
        snac_codebook_size: int = 4096,
        snac_n_levels: int = 3,
    ):
        self.phoneme_list = list(phoneme_list)
        self.stress_markers = list(stress_markers or [])
        self.snac_codebook_size = snac_codebook_size
        self.snac_n_levels = snac_n_levels

        # ── Build ID mappings ───────────────────────────────────────────
        next_id = NUM_SPECIAL

        # Linguistic tokens
        self._ling2id: dict[str, int] = {}
        for token in self.phoneme_list + self.stress_markers:
            self._ling2id[token] = next_id
            next_id += 1
        self._id2ling: dict[int, str] = {v: k for k, v in self._ling2id.items()}
        self.n_ling = len(self._ling2id)

        # Prosodic plan tokens
        self._pros2id: dict[str, int] = {}
        for token in ALL_PROSODY_TOKENS:
            self._pros2id[token] = next_id
            next_id += 1
        self._id2pros: dict[int, str] = {v: k for k, v in self._pros2id.items()}
        self.n_pros = len(self._pros2id)

        # Audio tokens
        self.audio_offset = next_id
        self.n_audio = snac_n_levels * snac_codebook_size

    @property
    def vocab_size(self) -> int:
        return self.audio_offset + self.n_audio

    @property
    def pad_id(self) -> int:
        return PAD

    # ── Encoding ────────────────────────────────────────────────────────

    def encode_phoneme(self, phone: str) -> int:
        return self._ling2id.get(phone, UNK)

    def encode_phonemes(self, phones: List[str]) -> List[int]:
        return [self.encode_phoneme(p) for p in phones]

    def encode_prosody(self, token: str) -> int:
        return self._pros2id.get(token, UNK)

    def encode_prosody_seq(self, tokens: List[str]) -> List[int]:
        return [self.encode_prosody(t) for t in tokens]

    def encode_snac(self, level: int, code: int) -> int:
        """Encode a single SNAC code at a given level to a unified token ID."""
        return self.audio_offset + level * self.snac_codebook_size + code

    def encode_snac_flat(self, codes_flat: List[tuple[int, int]]) -> List[int]:
        """Encode a flat sequence of (level, code) pairs."""
        return [self.encode_snac(level, code) for level, code in codes_flat]

    # ── Decoding ────────────────────────────────────────────────────────

    def decode_snac(self, token_id: int) -> tuple[int, int]:
        """Decode a unified token ID to (level, code)."""
        offset = token_id - self.audio_offset
        level = offset // self.snac_codebook_size
        code = offset % self.snac_codebook_size
        return level, code

    def is_audio_token(self, token_id: int) -> bool:
        return token_id >= self.audio_offset

    def is_prosody_token(self, token_id: int) -> bool:
        pros_start = NUM_SPECIAL + self.n_ling
        return pros_start <= token_id < pros_start + self.n_pros

    def is_linguistic_token(self, token_id: int) -> bool:
        return NUM_SPECIAL <= token_id < NUM_SPECIAL + self.n_ling

    # ── Sequence building ───────────────────────────────────────────────

    def build_training_sequence(
        self,
        ling_ids: List[int],
        prosody_ids: List[int],
        audio_ids: List[int],
    ) -> dict:
        """Build a three-phase training sequence.

        Returns:
            dict with input_ids, labels, phase boundaries
        """
        # [BOS] ling... [PLAN] prosody... [AUDIO] audio... [EOS]
        input_ids = (
            [BOS]
            + ling_ids
            + [PLAN_START]
            + prosody_ids
            + [AUDIO_START]
            + audio_ids
            + [EOS]
        )

        # Prefix = BOS + ling + PLAN_START (no loss)
        prefix_len = 1 + len(ling_ids) + 1

        # Labels: shifted by 1, mask prefix with -100
        labels = [-100] * len(input_ids)
        for i in range(prefix_len - 1, len(input_ids) - 1):
            labels[i] = input_ids[i + 1]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prefix_len": prefix_len,
            "plan_len": len(prosody_ids),
            "audio_len": len(audio_ids),
        }

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "phoneme_list": self.phoneme_list,
            "stress_markers": self.stress_markers,
            "snac_codebook_size": self.snac_codebook_size,
            "snac_n_levels": self.snac_n_levels,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CGNv2Tokenizer:
        path = Path(path)
        data = json.loads(path.read_text())
        return cls(
            phoneme_list=data["phoneme_list"],
            stress_markers=data.get("stress_markers", []),
            snac_codebook_size=data.get("snac_codebook_size", 4096),
            snac_n_levels=data.get("snac_n_levels", 3),
        )
