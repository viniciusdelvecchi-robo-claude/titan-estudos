import time as _time
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import pytest

from titan.calendario import servidor_para_utc
from titan.gramatica import (agregar_h1, conferir_causalidade_lote, gerar, regras_do_ativo)
from titan.tipos import Sessao
from titan.tipos_f2 import Lado
from titan.visao import Barras, ns

INICIO = date(2024, 1, 2)


def _dias(n):
    d, out = INICIO, []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _dados(n_dias, semente, base=1_100_000, simbolo="EURUSD"):
    dias = _dias(n_dias)
    ts = np.array([ns(datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(minutes=15 * k))
                   for d in dias for k in range(96)], dtype=np.int64)
    rng = np.random.default_rng(semente)
    c = base + np.cumsum(rng.integers(-300, 301, size=len(ts)))
    o = np.r_[base, c[:-1]]
    h = np.maximum(o, c) + rng.integers(0, 80, size=len(ts))
    l = np.minimum(o, c) - rng.integers(0, 80, size=len(ts))
    m15 = Barras("M15", ts, o, h, l, c, np.full(len(ts), 8, np.int64))
    sess = {}
    for d in dias:
        a = datetime(d.year, d.month, d.day, tzinfo=UTC)
        sess[d] = Sessao(simbolo, d, a, a + timedelta(days=1), a + timedelta(hours=23, minutes=45),
                         a + timedelta(days=1), 0, False, "sintetico")
    return m15, sess


def test_tamanho_da_gramatica():
    assert len(regras_do_ativo("EURUSD")) == 32 and len(regras_do_ativo("Ger40")) == 45  # B6 = Núcleo


def test_b6_e_o_nucleo_literal():
    from titan.regras import Nucleo, janela_nucleo
    from titan.tipos_f2 import Campanha
    from titan.visao import VisaoMemoria
    m15, sess = _dados(100, 9, base=1_800_000, simbolo="Ger40")
    b6 = sorted(i.intencao_id for i in gerar("Ger40", m15, sess) if i.regra_id == Nucleo.regra_id)
    vis = VisaoMemoria({("Ger40", "M15"): m15}, datetime(2024, 1, 2, tzinfo=UTC),
                       universo={Campanha.DAY_TRADE_M15: ("Ger40",)})
    nuc = Nucleo(simbolos=("Ger40",))
    ref = sorted(i.intencao_id for d in sorted(sess)
                 for i in nuc.decidir(vis.em(janela_nucleo("Ger40", d)[1]), janela_nucleo("Ger40", d)[1]))
    assert b6 == ref and len(ref) > 30


def test_h1_so_horas_completas():
    m15, _ = _dados(2, 1)
    m15 = m15.fatia(0, 95)  # última hora com 3 barras
    h1 = agregar_h1(m15)
    assert len(h1) == 23
    assert h1.h[0] == m15.h[:4].max() and h1.c[0] == m15.c[3]


@pytest.mark.parametrize("simbolo", ["EURUSD", "Ger40"])
def test_familias_presentes_e_ids_declarados(simbolo):
    m15, sess = _dados(140, 2, simbolo=simbolo)
    it = gerar(simbolo, m15, sess)
    regras = Counter(i.regra_id for i in it)
    assert set(regras) <= set(regras_do_ativo(simbolo))
    esperado = {"B1", "B2", "B3", "B4", "B5"} | ({"nu"} if simbolo == "Ger40" else set())  # B6 = Núcleo
    assert {r[:2] for r in regras} == esperado
    if simbolo == "Ger40":
        assert any("|avista|" in r for r in regras)


def test_zeragem_antes_de_l_e_da_hora_23_do_servidor():
    m15, sess = _dados(60, 3)
    for i in gerar("EURUSD", m15, sess):
        d = i.barra_sinal_fim.date()
        teto = servidor_para_utc(datetime.combine(d, time(22, 45)))
        assert i.hora_limite <= max(teto, servidor_para_utc(datetime.combine(d + timedelta(days=1), time(22, 45))))
        assert i.hora_limite - i.barra_sinal_fim >= timedelta(minutes=15)


def test_b1_lado_segue_a_formacao():
    m15, sess = _dados(30, 4)
    c = m15.c.copy()
    dia = 96 * 10  # 11º dia: força alta forte entre a abertura e +30 min
    c[dia: dia + 2] = c[dia - 1] + 50_000
    c[dia + 2:] += 50_000
    o = np.r_[c[0], c[:-1]]
    m = Barras("M15", m15.ts, o, np.maximum(o, c) + 1, np.minimum(o, c) - 1, c, m15.spread)
    b1 = [i for i in gerar("EURUSD", m, sess) if i.regra_id.startswith("B1|sessao|f30")
          and i.barra_sinal_fim.date() == _dias(30)[10]]
    assert b1 and all(i.lado is Lado.COMPRA for i in b1)


@pytest.mark.parametrize("semente", [11, 12])
def test_b3_vetorizada_igual_a_referencia(semente):
    from titan.gramatica import _Serie, b3_sazonal, b3_sazonal_ref, referencias
    m15, sess = _dados(160, semente)
    rng = np.random.default_rng(semente)
    keep = np.sort(rng.choice(len(m15), size=int(len(m15) * 0.97), replace=False))  # buracos esparsos
    m = Barras("M15", m15.ts[keep], m15.o[keep], m15.h[keep], m15.l[keep], m15.c[keep], m15.spread[keep])
    refs, s = referencias("EURUSD", sess), _Serie(m)
    nova = sorted(b.intencao().intencao_id + b.lado for b in b3_sazonal("EURUSD", s, refs))
    ref = sorted(b.intencao().intencao_id + b.lado for b in b3_sazonal_ref("EURUSD", s, refs))
    assert nova == ref and len(ref) > 1000


def test_b4_salto_depois_de_buraco_nao_vira_sinal():
    m15, sess = _dados(30, 7)
    j = 96 * 25 + 40  # dia 26, 10:00
    keep = np.r_[np.arange(j - 4), np.arange(j, len(m15))]  # buraco de 4 barras antes de j
    c = m15.c[keep].copy()
    c[j - 4:] += 40_000  # salto enorme entre a última barra antes do buraco e a primeira depois
    o = m15.o[keep].copy()
    o[j - 4:] += 40_000
    m = Barras("M15", m15.ts[keep], o, np.maximum(o, c) + 1, np.minimum(o, c) - 1, c, m15.spread[keep])
    alvo = m.ts[j - 4] + 15 * 60 * 10**9  # fim da barra logo depois do buraco
    b4 = [i for i in gerar("EURUSD", m, sess) if i.regra_id.startswith("B4|M15") and ns(i.barra_sinal_fim) == alvo]
    assert b4 == []


@pytest.mark.parametrize("simbolo", ["EURUSD", "Ger40"])
def test_f2_2_causalidade_lote(simbolo):
    # 25 cortes na suíte; os 200 do plano rodam em ferramentas/medir_causalidade_gramatica.py
    m15, sess = _dados(140, 5, simbolo=simbolo)
    assert conferir_causalidade_lote(simbolo, m15, sess, n_cortes=25) == []


def test_f2_2_detector_pega_gramatica_que_olha_o_futuro():
    m15, sess = _dados(60, 6)

    def trapaceira(simbolo, m, s):  # escolhe o lado pelo fechamento da barra seguinte ao sinal
        out = []
        for i in gerar(simbolo, m, s):
            j = int(np.searchsorted(m.ts, ns(i.barra_sinal_fim)))
            if j < len(m):
                lado = Lado.COMPRA if m.c[j] > m.o[j] else Lado.VENDA
                out.append(type(i)(i.campanha, i.regra_id, i.simbolo, lado, i.barra_sinal_fim, i.stop_pontos,
                                   i.hora_limite))
        return out
    assert conferir_causalidade_lote("EURUSD", m15, sess, n_cortes=20, gerador=trapaceira)
