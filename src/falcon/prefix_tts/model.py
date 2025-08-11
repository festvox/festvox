"""
Model definition for prefix language‑model based TTS.

The `TTSPrefixLM` class implements a simple causal Transformer that jointly
models sequences of text/phoneme tokens and audio tokens.  Given a prefix of
text tokens, the model can autoregressively generate audio tokens.  The
architecture is deliberately minimalistic to ease experimentation and
export to ONNX.

Usage:

```python
from tts_lm.model import TTSPrefixLM
import torch

vocab_size = 1024  # total number of text + audio tokens
model = TTSPrefixLM(vocab_size=vocab_size, d_model=512, n_layers=8, n_heads=8)

input_ids = torch.tensor([[1, 2, 3, 4]])
logits = model(input_ids)
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TTSPrefixLMConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1


class TTSPrefixLM(nn.Module):
    """Causal Transformer model for prefix‑LM TTS."""

    def __init__(self, config: TTSPrefixLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
        )
        # We use a standard TransformerEncoder to process the sequence; causal masking
        # will be provided in the forward method.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return an upper‑triangular causal mask for sequence length `seq_len`."""
        return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute logits for the next token prediction.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len) containing token IDs.
            attention_mask: Optional mask tensor of shape (batch_size, seq_len).  Non‑zero
                entries indicate tokens that should be attended to (1) and zero entries
                are masked out.  If provided, the mask is combined with the causal
                mask to form the full attention mask.

        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        # Token embeddings
        x = self.embeddings(input_ids)  # (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # transformer expects (seq_len, batch, d_model)

        # Build causal mask
        causal_mask = self._causal_mask(seq_len, device)
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) → (seq_len, seq_len) per batch broadcast
            attn_mask = (1.0 - attention_mask.to(dtype=x.dtype)).masked_fill(
                (attention_mask == 0), float('-inf')
            )
            # Expand to (batch, seq_len, seq_len)
            attn_mask = attn_mask[:, None, :] + causal_mask
        else:
            attn_mask = causal_mask

        # Apply Transformer
        # The transformer encoder takes mask of shape (seq_len, seq_len)
        # We'll broadcast attn_mask over batch dimension
        if attention_mask is not None:
            # Combine causal mask and attention mask for each element in batch
            # Flatten batch dimension into the mask; TransformerEncoder does not support per
            # example masks directly, so we loop.  For efficiency you may want to
            # implement a custom forward in future.
            outputs = []
            for b in range(batch_size):
                mask_b = attn_mask[b]
                out_b = self.transformer(x[:, b:b+1, :], mask=mask_b)
                outputs.append(out_b)
            x_out = torch.cat(outputs, dim=1)
        else:
            x_out = self.transformer(x, mask=attn_mask)

        # Final layernorm
        x_out = self.ln_f(x_out)
        x_out = x_out.transpose(0, 1)  # (batch, seq_len, d_model)
        logits = self.output_proj(x_out)  # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens given a prefix.

        Args:
            prefix_ids: LongTensor of shape (batch_size, prefix_len)
            max_new_tokens: Maximum number of tokens to generate
            eos_token_id: If specified, stop generation when this token is produced
            temperature: Sampling temperature
            top_k: If provided, restrict sampling to the top_k logits
            top_p: If provided, perform nucleus sampling with threshold top_p

        Returns:
            generated: LongTensor of shape (batch_size, prefix_len + max_new_tokens)
        """
        device = prefix_ids.device
        generated = prefix_ids.clone()
        batch_size = prefix_ids.size(0)

        for _ in range(max_new_tokens):
            # Compute logits for the last position
            logits = self.forward(generated)  # (batch, seq_len, vocab)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(next_logits, top_k)
                probs = torch.softmax(top_k_values, dim=-1)
                next_token = top_k_indices.gather(-1, torch.multinomial(probs, num_samples=1))
            elif top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # Mask tokens with cumulative probs > top_p
                sorted_indices_to_keep = cumulative_probs <= top_p
                # Ensure at least one token is kept
                sorted_indices_to_keep[..., 0] = True
                filter_mask = ~sorted_indices_to_keep
                next_logits_filtered = next_logits.clone().masked_fill(
                    sorted_indices[filter_mask], float('-inf')
                )
                probs = torch.softmax(next_logits_filtered, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None:
                # Stop if all sequences in the batch have produced EOS
                if ((next_token == eos_token_id).all()):
                    break
        return generated