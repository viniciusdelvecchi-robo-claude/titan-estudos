"""F2-9: certificação da taxa de aprovação sem vantagem pelo limite superior exato de Clopper e Pearson.

Limite unilateral superior (confiança c) para k aprovações em n séries: o p tal que P(Bin(n, p) <= k) = 1 − c.
Com k = n, o limite é 1. Python puro (soma binomial exata em log + bisseção), sem dependência nova.
"""

from __future__ import annotations

import math


def binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    termos = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
              for i in range(k + 1)]
    m = max(termos)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in termos))


def limite_superior(k: int, n: int, confianca: float = 0.95) -> float:
    if not 0 <= k <= n or n < 1:
        raise ValueError("precisa 0 <= k <= n e n >= 1")
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - (1.0 - confianca) ** (1.0 / n)
    lo, hi = k / n, 1.0
    for _ in range(200):  # cdf decresce em p
        meio = (lo + hi) / 2
        lo, hi = (meio, hi) if binom_cdf(k, n, meio) > 1 - confianca else (lo, meio)
        if hi - lo < 1e-15:
            break
    return hi


def n_minimo(teto: float, k_admitido: int = 0, confianca: float = 0.95) -> int:
    """Menor n em que k_admitido aprovações ainda deixam o limite superior <= teto."""
    n = max(k_admitido + 1, math.ceil(math.log(1 - confianca) / math.log(1 - teto)))
    while limite_superior(k_admitido, n, confianca) > teto:
        n += 1
    return n


def certifica(k: int, n: int, teto: float = 0.025, confianca: float = 0.95) -> bool:
    return limite_superior(k, n, confianca) <= teto
