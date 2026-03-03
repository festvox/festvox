"""CGN v2 — Prosodic Chain-of-Thought TTS.

Autoregressive transformer that generates an explicit prosodic plan
(duration, pitch contour, emphasis, pauses) before generating audio tokens.
"""

from .config import CGNv2Config
from .model import CGNv2
from .tokenizer import CGNv2Tokenizer
