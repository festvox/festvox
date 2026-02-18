"""CGN v2 transformer — Llama-style AR model with three-phase sequence.

Phase 1: Linguistic prefix (phonemes + stress, no loss)
Phase 2: Prosodic plan (duration, pitch, emphasis, pauses)
Phase 3: Audio tokens (SNAC codec, flattened multi-scale)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CGNv2Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        if offset + seq_len > self.cos_cached.shape[0]:
            self._build_cache(offset + seq_len)
        return (
            self.cos_cached[offset : offset + seq_len],
            self.sin_cached[offset : offset + seq_len],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos.unsqueeze(0).unsqueeze(0) + _rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, config: CGNv2Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal and kv_cache is None, dropout_p=dropout_p,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), new_kv


class FeedForward(nn.Module):
    def __init__(self, config: CGNv2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: CGNv2Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, new_kv = self.attn(self.attn_norm(x), cos, sin, kv_cache, is_causal)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, new_kv


class CGNv2(nn.Module):
    """CGN v2 — Prosodic Chain-of-Thought TTS transformer.

    Three-phase autoregressive model:
      Phase 1: Linguistic prefix (conditioning, no loss)
      Phase 2: Prosodic plan generation
      Phase 3: Audio token generation
    """

    def __init__(self, config: CGNv2Config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.output_proj.weight = self.token_embedding.weight

        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable/disable gradient checkpointing for memory savings."""
        self._gradient_checkpointing = enable

    @property
    def gradient_checkpointing(self) -> bool:
        return getattr(self, "_gradient_checkpointing", False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits [B, S, vocab_size]."""
        B, S = input_ids.shape
        x = self.dropout(self.token_embedding(input_ids))

        cos, sin = self.rope(S)
        cos, sin = cos.to(x.device), sin.to(x.device)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, None, True,
                    use_reentrant=False,
                )
            else:
                x, _ = layer(x, cos, sin, kv_cache=None, is_causal=True)

        return self.output_proj(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_tokens: Optional[set[int]] = None,
        cfg_scale: float = 1.0,
        uncond_input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV-cache and optional CFG.

        Args:
            input_ids: [1, S] prompt tokens
            max_new_tokens: max tokens to generate
            temperature, top_k, top_p: sampling parameters
            stop_tokens: set of token IDs that stop generation
            cfg_scale: classifier-free guidance scale (1.0 = no guidance)
            uncond_input_ids: [1, S'] unconditional prompt for CFG

        Returns:
            [1, S + generated] full sequence
        """
        self.eval()
        B, S = input_ids.shape
        assert B == 1
        device = input_ids.device

        if stop_tokens is None:
            from .tokenizer import EOS
            stop_tokens = {EOS}

        use_cfg = cfg_scale > 1.0 and uncond_input_ids is not None
        generated = input_ids

        # ── Prefill ─────────────────────────────────────────────────
        x = self.token_embedding(generated)
        cos, sin = self.rope(S)
        cos, sin = cos.to(device), sin.to(device)

        kv_caches = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, kv_caches[i] = layer(x, cos, sin, kv_cache=None, is_causal=True)
        logits = self.output_proj(self.norm(x))[:, -1:, :]

        # CFG prefill (unconditional)
        if use_cfg:
            ux = self.token_embedding(uncond_input_ids)
            uS = uncond_input_ids.shape[1]
            ucos, usin = self.rope(uS)
            ucos, usin = ucos.to(device), usin.to(device)
            ukv_caches = [None] * len(self.layers)
            for i, layer in enumerate(self.layers):
                ux, ukv_caches[i] = layer(ux, ucos, usin, kv_cache=None, is_causal=True)
            uncond_logits = self.output_proj(self.norm(ux))[:, -1:, :]
            logits = uncond_logits + cfg_scale * (logits - uncond_logits)

        next_token = _sample(logits[:, -1, :], temperature, top_k, top_p)
        generated = torch.cat([generated, next_token], dim=1)

        # ── Decode loop ─────────────────────────────────────────────
        for step in range(1, max_new_tokens):
            if next_token.item() in stop_tokens:
                break

            cur_pos = S + step - 1
            x = self.token_embedding(next_token)
            cos, sin = self.rope(1, offset=cur_pos)
            cos, sin = cos.to(device), sin.to(device)

            for i, layer in enumerate(self.layers):
                x, kv_caches[i] = layer(x, cos, sin, kv_cache=kv_caches[i], is_causal=False)
            logits = self.output_proj(self.norm(x))

            if use_cfg:
                # Unconditional step
                ucur_pos = uncond_input_ids.shape[1] + step - 1
                ux = self.token_embedding(next_token)
                ucos, usin = self.rope(1, offset=ucur_pos)
                ucos, usin = ucos.to(device), usin.to(device)
                for i, layer in enumerate(self.layers):
                    ux, ukv_caches[i] = layer(ux, ucos, usin, kv_cache=ukv_caches[i], is_causal=False)
                uncond_logits = self.output_proj(self.norm(ux))
                logits = uncond_logits + cfg_scale * (logits - uncond_logits)

            next_token = _sample(logits[:, -1, :], temperature, top_k, top_p)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
