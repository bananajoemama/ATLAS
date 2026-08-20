"""
Deterministic seed generation.

Python's built-in hash() randomizes string hashing per-process (PYTHONHASHSEED)
for security reasons, so hash(('label', i)) gives DIFFERENT values across
separate runs of the same script. This silently breaks reproducibility of
Monte Carlo results -- two runs of the same experiment produce different
(though statistically similar) numbers, which is not acceptable for
results reported in a paper.

This module provides a deterministic alternative using hashlib, whose
output is stable across processes, machines, and Python versions.
"""
import hashlib


def deterministic_seed(key):
    """
    Convert an arbitrary (hashable-by-repr) key into a deterministic
    32-bit unsigned integer seed, stable across runs/processes.
    """
    key_bytes = repr(key).encode('utf-8')
    digest = hashlib.sha256(key_bytes).digest()
    return int.from_bytes(digest[:4], byteorder='big')
