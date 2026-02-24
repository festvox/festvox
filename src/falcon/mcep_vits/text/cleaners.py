"""Text cleaners: normalize and optionally phonemize input text.

Cleaners are selected by name in the config JSON (e.g. "english_cleaners2").
"""

import re
from unidecode import unidecode
from phonemizer import phonemize

_whitespace_re = re.compile(r'\s+')

_abbreviations = [
    (re.compile(r'\b%s\.' % abbr, re.IGNORECASE), expansion)
    for abbr, expansion in [
        ('mrs', 'misess'), ('mr', 'mister'), ('dr', 'doctor'),
        ('st', 'saint'), ('co', 'company'), ('jr', 'junior'),
        ('maj', 'major'), ('gen', 'general'), ('drs', 'doctors'),
        ('rev', 'reverend'), ('lt', 'lieutenant'), ('hon', 'honorable'),
        ('sgt', 'sergeant'), ('capt', 'captain'), ('esq', 'esquire'),
        ('ltd', 'limited'), ('col', 'colonel'), ('ft', 'fort'),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def basic_cleaners(text):
    """Lowercase and collapse whitespace."""
    return collapse_whitespace(text.lower())


def transliteration_cleaners(text):
    """Transliterate to ASCII, lowercase, collapse whitespace."""
    return collapse_whitespace(unidecode(text).lower())


def english_cleaners(text):
    """ASCII + lowercase + abbreviations + espeak phonemization."""
    text = expand_abbreviations(unidecode(text).lower())
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
    return collapse_whitespace(phonemes)


def english_cleaners2(text):
    """Same as english_cleaners but preserves punctuation and stress marks."""
    text = expand_abbreviations(unidecode(text).lower())
    phonemes = phonemize(
        text, language='en-us', backend='espeak', strip=True,
        preserve_punctuation=True, with_stress=True,
    )
    return collapse_whitespace(phonemes)
