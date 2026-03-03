"""Text processing: symbol-to-ID mapping and text cleaning."""

from text import cleaners
from text.symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def _clean_text(text, cleaner_names):
    """Apply cleaners to text string."""
    for name in cleaner_names:
        text = getattr(cleaners, name)(text)
    return text


def text_to_sequence(text, cleaner_names):
    """Convert text string to a list of symbol IDs, applying cleaners first."""
    cleaned = _clean_text(text, cleaner_names)
    return cleaned_text_to_sequence(cleaned)


def cleaned_text_to_sequence(cleaned_text):
    """Convert pre-cleaned text (space-delimited phones) to a list of symbol IDs."""
    return [_symbol_to_id[p] for p in cleaned_text.split() if p in _symbol_to_id]


def sequence_to_text(sequence):
    """Convert list of symbol IDs back to a space-delimited string."""
    return ' '.join(_id_to_symbol[i] for i in sequence)
