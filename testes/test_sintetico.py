import numpy as np
import pytest

from titan.sintetico import BARRAS_DIA, Vantagem, gerar_mundo, sharpe_ideal


def test_deterministico_e_consistente():
    a = gerar_mundo(3, 40, 3)
    b = gerar_mundo(3, 40, 3)
    for s in a.m15:
        x, y = a.m15[s], b.m15[s]
        assert np.array_equal(x.c, y.c) and np.array_equal(x.ts, y.ts)
        assert len(x) == 40 * BARRAS_DIA
        assert (x.l <= np.minimum(x.o, x.c)).all() and (x.h >= np.maximum(x.o, x.c)).all()
    assert len(a.barras_motor("SINT00")) == 40 * BARRAS_DIA
    assert a.com_vantagem == ()


def test_vantagem_so_na_fracao_e_mesmos_choques():
    base = gerar_mundo(4, 60, 4)
    w = gerar_mundo(4, 60, 4, Vantagem("sazonal", 0.5, 0.5))
    assert w.com_vantagem == ("SINT00", "SINT01")
    assert np.array_equal(w.m15["SINT03"].c, base.m15["SINT03"].c)  # sem vantagem: idêntico
    assert not np.array_equal(w.m15["SINT00"].c, base.m15["SINT00"].c)


def test_sharpe_ideal_cresce_com_a_forca_e_some_sem_vantagem():
    fracos = sharpe_ideal(gerar_mundo(5, 600, 4, Vantagem("momento", 0.05)), Vantagem("momento", 0.05))
    fortes = sharpe_ideal(gerar_mundo(5, 600, 4, Vantagem("momento", 0.2)), Vantagem("momento", 0.2))
    assert fortes > fracos + 1.0
    zero = gerar_mundo(5, 600, 4, Vantagem("momento", 0.0))
    assert sharpe_ideal(zero, Vantagem("momento", 0.0)) < 0  # só paga spread


def test_vantagem_invalida():
    with pytest.raises(ValueError):
        Vantagem("magica", 1.0)
    with pytest.raises(ValueError):
        Vantagem("nenhuma", 0.1)
