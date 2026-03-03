"""Symbol set for ARPAbet phoneme input (flite G2P)."""

_pad = '_'
_phones = [
    "aa", "ae", "ah", "ao", "aw", "ax", "ay",
    "b", "ch", "d", "dh", "eh", "er", "ey",
    "f", "g", "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng",
    "ow", "oy", "p", "pau", "r", "s", "sh", "t", "th",
    "uh", "uw", "v", "w", "y", "z", "zh",
]

symbols = [_pad] + _phones  # 42 total
