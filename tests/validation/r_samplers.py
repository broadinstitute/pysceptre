"""Exact replicas of two of sceptre's permutation samplers. Test-only.

``fisher_yates_samlper`` and ``hybrid_fisher_iwor_sampler`` reproduce the C++
functions of the same names in sceptre 0.10.3
(``src/generate_samples_functions.cpp``) draw for draw and entry for entry:
row ``j`` of the result is draw ``j`` as ``sceptre:::synth_idx_list_to_r_list``
returns it, 0-based and in R's order. They let a test hand R's own permutation
draws to pysceptre, so permutation p-values can be compared with R value for
value.

Each call of either C++ sampler seeds a fresh ``boost::random::mt19937`` with
4 and turns every 32-bit word into one uniform, ``word / 2**32``. A draw
consumes exactly ``M`` words, so draw ``j`` reads words ``j * M`` to
``(j + 1) * M - 1``, and the replicas work through the draws in chunks.

This module lives under ``tests/`` and is not part of the pysceptre package.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

# Working-memory budget per chunk of draws, in array elements.
_CHUNK_ELEMENTS = 1 << 22

_MT_STATE_WORDS = 624


def boost_mt19937(seed: int = 4) -> np.random.MT19937:
    """Return a numpy ``MT19937`` in the state of ``boost::random::mt19937(seed)``.

    boost seeds with the reference ``init_genrand`` recurrence, which numpy's
    ``MT19937(seed)`` and ``default_rng(seed)`` do not reproduce. The key is
    built here and installed through the public ``state`` setter, so
    ``random_raw`` on the result returns boost's 32-bit words in order.
    """
    key = np.empty(_MT_STATE_WORDS, dtype=np.uint32)
    word = seed & 0xFFFFFFFF
    for i in range(_MT_STATE_WORDS):
        key[i] = word
        word = (1812433253 * (word ^ (word >> 30)) + i + 1) & 0xFFFFFFFF
    bitgen = np.random.MT19937(0)
    bitgen.state = {"bit_generator": "MT19937", "state": {"key": key, "pos": _MT_STATE_WORDS}}
    return bitgen


def _uniform_chunks(M: int, B: int, rows: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start, u)`` for consecutive chunks of at most ``rows`` draws.

    ``u`` is the ``(n, M)`` float64 array of the boost ``uniform_real(0, 1)``
    values that draws ``start`` to ``start + n - 1`` consume, in order.
    """
    bitgen = boost_mt19937(4)
    for start in range(0, B, rows):
        n = min(rows, B - start)
        words = bitgen.random_raw(n * M)
        yield start, (words.astype(np.float64) / 2.0**32).reshape(n, M)


def fisher_yates_samlper(n_tot: int, M: int, B: int) -> np.ndarray:
    """Replicate ``sceptre:::fisher_yates_samlper(n_tot, M, B)`` exactly.

    Each draw makes ``M`` Fisher-Yates swaps on ``0, ..., n_tot - 1``, each
    moving a uniformly chosen element to the tail, and returns the ``M`` tail
    elements, last pick first. The name keeps R's spelling.

    Requires ``0 <= M <= n_tot``. Returns a ``(B, M)`` int64 array whose row
    ``j`` is R's draw ``j``.
    """
    if B < 0 or M < 0 or M > n_tot:
        raise ValueError(f"need B >= 0 and 0 <= M <= n_tot; got n_tot={n_tot}, M={M}, B={B}")
    out = np.empty((B, M), dtype=np.int64)
    if B == 0 or M == 0:
        return out
    rows = max(1, _CHUNK_ELEMENTS // (n_tot + M))
    scale = (n_tot - np.arange(M)).astype(np.float64)
    dst = n_tot - 1 - np.arange(M)
    for start, u in _uniform_chunks(M, B, rows):
        n = u.shape[0]
        base = np.arange(n, dtype=np.int64) * n_tot
        src = (np.floor(scale * u).astype(np.int64) + base[:, None]).T.copy()
        x = np.tile(np.arange(n_tot, dtype=np.int32), n)
        for i in range(M):
            s, d = src[i], base + dst[i]
            picked = x[s]
            x[s] = x[d]
            x[d] = picked
        out[start : start + n] = x.reshape(n, n_tot)[:, n_tot - M :]
    return out


# R's name carries a typo; the corrected spelling works too.
fisher_yates_sampler = fisher_yates_samlper


def hybrid_fisher_iwor_sampler(N: int, m: int, M: int, B: int) -> np.ndarray:
    """Replicate ``sceptre:::hybrid_fisher_iwor_sampler(N, m, M, B)`` exactly.

    Each draw makes ``m`` Fisher-Yates swaps on ``0, ..., N + m - 1``, whose
    picks fill entries ``0`` to ``m - 1`` last pick first, then extends the
    sample one entry at a time for sizes ``i = m + 1`` to ``M``: entry
    ``i - 1`` is the new element ``N + i - 1`` with probability
    ``i / (N + i)`` and otherwise a uniformly chosen element not yet picked.
    For every ``k`` from ``m`` to ``M``, the first ``k`` entries of a draw are
    ``k`` distinct elements of ``0, ..., N + k - 1``.

    Requires ``N >= 1`` and ``1 <= m <= M``. Returns a ``(B, M)`` int64 array
    whose row ``j`` is R's draw ``j``.
    """
    if N < 1 or m < 1 or m > M or B < 0:
        raise ValueError(f"need N >= 1, 1 <= m <= M and B >= 0; got N={N}, m={m}, M={M}, B={B}")
    out = np.empty((B, M), dtype=np.int64)
    if B == 0:
        return out
    width = N + m
    rows = max(1, _CHUNK_ELEMENTS // (width + M))
    fy_scale = (width - np.arange(m)).astype(np.float64)
    fy_dst = width - 1 - np.arange(m)
    sizes = np.arange(m + 1, M + 1, dtype=np.float64)
    keep_old = 1.0 - sizes / (N + sizes)
    for start, u in _uniform_chunks(M, B, rows):
        n = u.shape[0]
        base = np.arange(n, dtype=np.int64) * width
        x = np.tile(np.arange(width, dtype=np.int32), n)
        fy_src = (np.floor(fy_scale * u[:, :m]).astype(np.int64) + base[:, None]).T.copy()
        for i in range(m):
            s, d = fy_src[i], base + fy_dst[i]
            picked = x[s]
            x[s] = x[d]
            x[d] = picked
        out[start : start + n, :m] = x.reshape(n, width)[:, N:]
        if M == m:
            continue
        # Same operation order as the C++: (u * N) / (1 - p).
        u_grow = u[:, m:]
        pos = np.where(u_grow > keep_old, N, np.floor(u_grow * N / keep_old)).astype(np.int64)
        grow_src = (pos + base[:, None]).T.copy()
        top = base + N
        x[top] = N + m
        grown = np.empty((M - m, n), dtype=np.int64)
        for k in range(M - m):
            s = grow_src[k]
            grown[k] = x[s]
            x[s] = x[top]
            x[top] = N + m + k + 1
        out[start : start + n, m:] = grown.T
    return out
