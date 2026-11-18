from datetime import date, timedelta

import numpy as np
import pytest

from titan.prova import (bloco_politis_white, lag_newey_west, p_bootstrap, t_newey_west)
from titan.prova import testar as portao
from titan.tipos_f2 import EntradaProva, ParamsPortao, ResultadoMotor

D0 = date(2021, 1, 4)


def _res(ret, blocos=None):
    datas = tuple(D0 + timedelta(days=i) for i in range(len(ret)))
    ret = tuple(float(x) for x in ret)
    pnl = tuple(x * 3000.0 for x in ret)
    if blocos is None:
        blocos = tuple(d.isoformat() for d in datas)
    z = (0.0,) * len(ret)
    return ResultadoMotor(datas, pnl, tuple(p / 3000.0 for p in pnl), z, z, tuple(blocos), (), (), (), ())


def _entrada(nome, ret, *, pess=None, blocos=None, cofre_em=None, rev=(False,) * 50, janela=0):
    n = len(ret)
    return EntradaProva(nome, _res(ret, blocos), _res(ret if pess is None else pess, blocos),
                        D0 + timedelta(days=janela), D0 + timedelta(days=cofre_em if cofre_em is not None else n),
                        rev)


def _params(fatia=0.025, B=499):
    return ParamsPortao(fatia=fatia, folga=0.8, passos_holm=2, apostas_min=30, nivel_cofre=0.05,
                        revogacao_max=0.2, reamostragens=B, semente=7)


def test_lag_newey_west():
    assert lag_newey_west(100) == 4 and lag_newey_west(1200) == 6


def test_tamanho_newey_west_sob_h0():
    rng = np.random.default_rng(1)
    rej = sum(t_newey_west(rng.standard_normal(500))[1] < 0.05 for _ in range(2000))
    assert 0.03 < rej / 2000 < 0.07


def test_bloco_pw_cresce_com_dependencia():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(1500)
    ar = np.zeros(1500)
    for i in range(1, 1500):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    assert bloco_politis_white(e) < 4 < bloco_politis_white(ar)


def test_bootstrap_rejeita_media_forte_e_nao_rejeita_media_negativa():
    rng = np.random.default_rng(3)
    assert p_bootstrap(0.3 + rng.standard_normal(800), 499, 1) < 0.01
    e = rng.standard_normal(800)
    assert p_bootstrap(e - e.mean() - 0.05, 499, 1) > 0.5


def test_bootstrap_tamanho_aproximado():
    rng = np.random.default_rng(4)
    rej = sum(p_bootstrap(rng.standard_normal(300), 199, i) < 0.10 for i in range(200))
    assert 0.04 < rej / 200 < 0.17


def test_f2_1_fatia_muda_o_critico():
    rng = np.random.default_rng(5)
    a = 0.0016 + 0.01 * rng.standard_normal(1200)  # p na casa de 0,5%
    b = 0.0016 + 0.01 * rng.standard_normal(1200)
    v1 = portao([_entrada("nucleo", a), _entrada("garimpo", b)], _params(0.025))
    v2 = portao([_entrada("nucleo", a), _entrada("garimpo", b)], _params(0.0025))
    assert v1.candidatos[0].nivel_aplicado == pytest.approx(0.010)
    assert v2.candidatos[0].nivel_aplicado == pytest.approx(0.001)
    assert any(c.aprovado for c in v1.candidatos) and not any(c.aprovado for c in v2.candidatos)


def test_holm_para_se_o_primeiro_reprova():
    rng = np.random.default_rng(6)
    v = portao([_entrada("a", 0.01 * rng.standard_normal(1200)), _entrada("b", 0.01 * rng.standard_normal(1200))],
               _params())
    assert v.candidatos[1].nivel_aplicado is None and "holm_parou" in v.candidatos[1].motivos


def test_pessimista_tambem_precisa_passar():
    rng = np.random.default_rng(7)
    e = rng.standard_normal(1200)
    a = 0.0012 + 0.01 * (e - e.mean())  # Sharpe ~1,9: abaixo do limiar T13, t ~4
    v = portao([_entrada("a", a, pess=a - 0.002), _entrada("b", a)], _params())
    c = {x.candidato: x for x in v.candidatos}
    assert not c["a"].aprovado and c["b"].aprovado


def test_f2_8_trinta_entradas_no_mesmo_dia_sao_um_bloco():
    rng = np.random.default_rng(8)
    a = 0.003 + 0.01 * rng.standard_normal(1200)
    blocos = ("2021-01-04",) * 30
    v = portao([_entrada("a", a, blocos=blocos), _entrada("b", a)], _params())
    c = {x.candidato: x for x in v.candidatos}
    assert c["a"].blocos_independentes == 1 and "blocos_insuficientes" in c["a"].motivos


def test_veto_do_cofre_e_sentinela():
    rng = np.random.default_rng(9)
    a = 0.003 + 0.01 * rng.standard_normal(1400)
    a[1200:] = -0.01 + 0.005 * rng.standard_normal(200)
    v = portao([_entrada("a", a, cofre_em=1200), _entrada("b", a[:1200], rev=(True,) * 11 + (False,) * 39)],
               _params())
    c = {x.candidato: x for x in v.candidatos}
    assert "veto_do_cofre" in c["a"].motivos
    assert "sentinela_revogaria" in c["b"].motivos


def test_f2_2_aquecimento_nao_entra():
    rng = np.random.default_rng(10)
    a = 0.002 + 0.01 * rng.standard_normal(1300)
    b = a.copy()
    b[:100] = 50.0  # lixo no aquecimento
    v1 = portao([_entrada("a", a, janela=100), _entrada("b", a, janela=100)], _params())
    v2 = portao([_entrada("a", b, janela=100), _entrada("b", b, janela=100)], _params())
    assert v1 == v2


def test_numero_de_candidatos_tem_de_bater_com_holm():
    with pytest.raises(ValueError):
        portao([_entrada("a", np.zeros(100))], _params())
