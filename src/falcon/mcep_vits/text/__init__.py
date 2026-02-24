"""Text processing: symbol-to-ID mapping and text cleaning."""

from text import cleaners
from text.symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
    """Convert text string to a list of symbol IDs, applying cleaners first."""
    for name in cleaner_names:
        text = getattr(cleaners, name)(text)
    return [_symbol_to_id[s] for s in text]


def cleaned_text_to_sequence(cleaned_text):
    """Convert pre-cleaned text string to a list of symbol IDs."""
    return [_symbol_to_id[s] for s in cleaned_text]


def sequence_to_text(sequence):
    """Convert list of symbol IDs back to a string."""
    return ''.join(_id_to_symbol[i] for i in sequence)
