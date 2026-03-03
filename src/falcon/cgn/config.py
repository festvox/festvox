"""CGN v2 model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CGNv2Config:
    """Configuration for CGN v2 transformer."""

    # Model dimensions
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048

    # Vocabulary (set by tokenizer)
    vocab_size: int = 12500

    # Sequence
    max_seq_len: int = 4096

    # Regularization
    dropout: float = 0.1

    # RoPE
    rope_theta: float = 10000.0

    # Weight tying
    weight_tying: bool = True

    # Normalization
    norm_eps: float = 1e-5

    # CFG: probability of dropping linguistic prefix during training
    cfg_drop_prob: float = 0.1

    # Codec (supports "snac" and "qwen12hz")
    codec_type: str = "snac"
    snac_n_levels: int = 3
    snac_codebook_size: int = 4096
    snac_sample_rate: int = 24000

    # Prosodic plan token counts
    n_duration_bins: int = 8
    n_pitch_contours: int = 8
    n_emphasis_levels: int = 4
    n_pause_types: int = 4

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_audio_tokens(self) -> int:
        return self.snac_n_levels * self.snac_codebook_size

    def param_count_estimate(self) -> int:
        """Rough parameter count estimate."""
        embed = self.vocab_size * self.d_model
        attn = self.n_layers * (
            self.d_model * self.n_heads * self.head_dim  # Q
            + self.d_model * self.n_kv_heads * self.head_dim * 2  # K, V
            + self.n_heads * self.head_dim * self.d_model  # O
        )
        ffn = self.n_layers * (3 * self.d_model * self.d_ff)  # gate, up, down
        norm = self.n_layers * 2 * self.d_model + self.d_model  # per-layer + final
        output = 0 if self.weight_tying else self.vocab_size * self.d_model
        return embed + attn + ffn + norm + output


# Size presets
CONFIGS = {
    "tiny": CGNv2Config(d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, d_ff=1024),
    "mini": CGNv2Config(d_model=512, n_layers=6, n_heads=8, n_kv_heads=2, d_ff=1024),
    "small": CGNv2Config(d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, d_ff=2048),
    "base": CGNv2Config(d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4, d_ff=3072),
}


def get_config(size: str = "small", **overrides) -> CGNv2Config:
    """Get a config by size name, with optional overrides."""
    if size not in CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(CONFIGS.keys())}")
    cfg = CONFIGS[size]
    if overrides:
        from dataclasses import asdict
        d = {**asdict(cfg), **overrides}
        return CGNv2Config(**d)
    return cfg
