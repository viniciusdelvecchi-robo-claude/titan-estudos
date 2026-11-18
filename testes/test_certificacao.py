import math

import pytest

from titan.certificacao import binom_cdf, certifica, limite_superior, n_minimo


def test_cdf_contra_soma_direta():
    for k, n, p in ((0, 10, 0.1), (3, 20, 0.2), (7, 50, 0.05)):
        direta = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
        assert binom_cdf(k, n, p) == pytest.approx(direta, rel=1e-12)


def test_zero_aprovacoes_tem_forma_fechada():
    for n in (10, 119, 500):
        assert limite_superior(0, n) == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-12)


def test_definicao_pela_binomial():
    for k, n in ((1, 200), (3, 400), (7, 1000)):
        p = limite_superior(k, n)
        assert binom_cdf(k, n, p) == pytest.approx(0.05, abs=1e-9)
    p = limite_superior(1, 200)  # conferência independente pela forma fechada de k = 1
    assert (1 - p) ** 200 + 200 * p * (1 - p) ** 199 == pytest.approx(0.05, abs=1e-12)


def test_n_minimo_campanha_1():
    assert n_minimo(0.025) == 119  # ln 0,05 / ln 0,975 = 118,3
    n1 = n_minimo(0.025, 1)
    assert limite_superior(1, n1) <= 0.025 < limite_superior(1, n1 - 1)
    assert certifica(0, 119) and not certifica(0, 118)


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        limite_superior(5, 3)
    assert limite_superior(4, 4) == 1.0
