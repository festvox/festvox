"""Text cleaners: normalize and phonemize input text.

Cleaners are selected by name in the config JSON (e.g. "flite_cleaners").
"""

import re
import subprocess
from unidecode import unidecode

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

_stress_re = re.compile(r'[0-9]')


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def _flite_g2p(text):
    """Run flite G2P, return space-delimited ARPAbet phones (stress stripped)."""
    try:
        result = subprocess.run(
            ["flite", "-t", text, "-o", "/dev/null", "-ps"],
            capture_output=True, text=True, timeout=60,
        )
        phones_str = result.stdout.strip()
        if not phones_str:
            return ""
        phones = []
        for p in phones_str.split():
            phones.append(_stress_re.sub('', p))
        return ' '.join(phones)
    except Exception:
        return ""


def flite_cleaners(text):
    """Expand abbreviations, transliterate to ASCII, run flite G2P."""
    text = expand_abbreviations(unidecode(text).lower())
    return _flite_g2p(text)


def basic_cleaners(text):
    """Lowercase and collapse whitespace."""
    return collapse_whitespace(text.lower())


def transliteration_cleaners(text):
    """Transliterate to ASCII, lowercase, collapse whitespace."""
    return collapse_whitespace(unidecode(text).lower())
