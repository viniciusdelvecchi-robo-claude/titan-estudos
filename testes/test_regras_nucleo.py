from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from titan.causalidade import conferir_causalidade
from titan.regras import Nucleo, janela_nucleo
from titan.tipos_f2 import Campanha, Lado
from titan.visao import Barras, FuturoRecusado, VisaoMemoria, ns

INICIO = date(2024, 1, 2)


def _serie(dias: int, semente: int, base: int = 500000) -> Barras:
    """M15 sintético, 24 h nos dias úteis (UTC), passeio aleatório em pontos."""
    ts = []
    d = INICIO
    while len(ts) < dias * 96:
        if d.weekday() < 5:
            t0 = datetime(d.year, d.month, d.day, tzinfo=UTC)
            ts.extend(ns(t0 + timedelta(minutes=15 * k)) for k in range(96))
        d += timedelta(days=1)
    ts = np.array(ts, dtype=np.int64)
    rng = np.random.default_rng(semente)
    c = base + np.cumsum(rng.integers(-200, 201, size=len(ts)))
    o = np.concatenate([[base], c[:-1]])
    h = np.maximum(o, c) + rng.integers(0, 50, size=len(ts))
    l = np.minimum(o, c) - rng.integers(0, 50, size=len(ts))
    return Barras("M15", ts, o, h, l, c, np.full(len(ts), 5, dtype=np.int64))


def _visao(dias=70):
    dados = {("Usa500", "M15"): _serie(dias, 1), ("Ger40", "M15"): _serie(dias, 2, 1_800_000)}
    return VisaoMemoria(dados, datetime(2024, 1, 2, tzinfo=UTC),
                        universo={Campanha.DAY_TRADE_M15: ("Usa500", "Ger40")})


def _entradas(dias=70):
    out = []
    d = INICIO
    for _ in range(dias * 7 // 5 + 3):
        for s in ("Usa500", "Ger40"):
            out.append(janela_nucleo(s, d)[1])
        d += timedelta(days=1)
    return sorted(set(out))


def test_janela_americana_recua_quinze_minutos_e_europeia_nao():
    a30, e, f, fech = janela_nucleo("Usa500", date(2024, 3, 1))  # inverno EUA: 16:00 NY = 21:00Z
    assert (fech, f, e, a30) == (datetime(2024, 3, 1, 21, tzinfo=UTC), datetime(2024, 3, 1, 20, 45, tzinfo=UTC),
                                 datetime(2024, 3, 1, 20, 15, tzinfo=UTC), datetime(2024, 3, 1, 15, tzinfo=UTC))
    _, e, f, fech = janela_nucleo("Ger40", date(2024, 7, 1))  # verão: 17:30 Berlim = 15:30Z
    assert f == fech == datetime(2024, 7, 1, 15, 30, tzinfo=UTC) and e == datetime(2024, 7, 1, 15, tzinfo=UTC)


def test_lado_segue_o_sinal_e_stop_vem_do_passado():
    v = _visao()
    n = Nucleo()
    vistos = 0
    for t in _entradas():
        for i in n.decidir(v.em(t), t):
            vistos += 1
            s = i.simbolo
            dia = t.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York" if s == "Usa500" else "Europe/Berlin")).date()
            a30, e, f, _ = janela_nucleo(s, dia)
            b = v.dados[(s, "M15")]
            fins = b.fim
            p30 = b.c[np.searchsorted(fins, ns(a30))]
            k = 1
            while True:
                fa = ns(janela_nucleo(s, dia - timedelta(days=k))[3])
                j = np.searchsorted(fins, fa)
                if j < len(fins) and fins[j] == fa:
                    break
                k += 1
            assert i.lado == (Lado.COMPRA if p30 > b.c[j] else Lado.VENDA)
            assert i.barra_sinal_fim == e and i.hora_limite == f and i.stop_pontos > 0
    assert vistos > 50  # só depois de 20 pregões de histórico para o stop


def test_sem_decisao_fora_do_instante_de_entrada():
    v = _visao()
    t = datetime(2024, 3, 5, 14, 0, tzinfo=UTC)
    assert Nucleo().decidir(v.em(t), t) == []


def test_visao_nao_mostra_barra_aberta():
    v = _visao(5)
    t = datetime(2024, 1, 3, 10, 7, tzinfo=UTC)
    b = v.em(t).barras("Usa500", "M15", 3)
    assert int(b.fim[-1]) <= ns(t)


def test_contrato_futuro_recusado():
    v = _visao(5)
    with pytest.raises(FuturoRecusado):
        v.contrato("Usa500", date(2024, 2, 1))


def test_f2_2_causalidade_200_cortes():
    v = _visao()
    divergencias = conferir_causalidade(Nucleo().decidir, v, _entradas(), n_cortes=200)
    assert divergencias == []


def test_f2_2_detector_pega_regra_que_olha_o_futuro():
    v = _visao(40)

    def trapaceira(visao, agora):  # escolhe o lado pela barra seguinte a agora
        out = []
        for i in Nucleo(simbolos=("Usa500",)).decidir(visao, agora):
            b = visao.em(agora + timedelta(minutes=15)).barras("Usa500", "M15", 1)
            if len(b) and b.fim[-1] > ns(agora):
                lado = Lado.COMPRA if b.c[-1] > b.o[-1] else Lado.VENDA
                out.append(type(i)(campanha=i.campanha, regra_id="trapaca", simbolo=i.simbolo, lado=lado,
                                   barra_sinal_fim=i.barra_sinal_fim, stop_pontos=i.stop_pontos,
                                   hora_limite=i.hora_limite))
        return out
    assert conferir_causalidade(trapaceira, v, _entradas(40), n_cortes=20)
