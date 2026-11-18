"""Comparação caixa-preta da reimplementação cega das partes B e C."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

import titan.aprendizagem as referencia_aprendizagem
import titan.gramatica as referencia_gramatica
from titan.cega_codex import aprendizagem as cega_aprendizagem
from titan.cega_codex import gramatica as cega_gramatica
from titan.sintetico import dias_uteis, gerar_mundo
from titan.tipos_f2 import Campanha, Intencao, Lado


def _chave_intencao(i):
    return (
        i.regra_id, i.lado, i.barra_sinal_fim, i.stop_pontos,
        i.alvo_pontos, i.hora_limite, i.intencao_id,
    )


def test_parte_b_cega_bate_todas_as_familias_campo_a_campo():
    mundo = gerar_mundo(77, 80, 1)
    args = ("SINT00", mundo.m15["SINT00"], mundo.sessoes["SINT00"])
    referencia = {_chave_intencao(i) for i in referencia_gramatica.gerar(*args)}
    cega = {_chave_intencao(i) for i in cega_gramatica.gerar(*args)}
    assert cega == referencia


def _normalizar_periodos(periodos):
    return [
        (p.inicio, p.fim, None if p.pesos is None else dict(p.pesos))
        for p in periodos
    ]


@pytest.mark.parametrize("variante", ["V0", "V1", "V2_20", "V2_50", "V3"])
def test_parte_c_cega_bate_pesos_por_periodo(variante):
    dias = dias_uteis(date(2019, 1, 2), 800)
    rng = np.random.default_rng(4)
    series = {
        (f"r{i}", f"S{i % 2}"): rng.normal((i - 1) * 0.0002, 0.01, len(dias))
        for i in range(4)
    }
    esperado = _normalizar_periodos(referencia_aprendizagem.agenda(
        variante, dias, series, reamostragens=49, semente=17,
    ))
    obtido = _normalizar_periodos(cega_aprendizagem.agenda(
        variante, dias, series, reamostragens=49, semente=17,
    ))
    assert [(a, b) for a, b, _ in obtido] == [(a, b) for a, b, _ in esperado]
    for (_, _, pe), (_, _, po) in zip(esperado, obtido):
        assert (pe is None) == (po is None)
        if pe is not None:
            assert set(pe) == set(po)
            for chave in pe:
                assert po[chave] == pytest.approx(pe[chave], abs=1e-12)


def test_parte_c_cega_aplicar_bate_id_lado_e_peso():
    dias = dias_uteis(date(2019, 1, 2), 800)
    x = np.linspace(-0.01, 0.03, len(dias))
    series = {("r", "X"): x, ("z", "Y"): -x}
    ref = referencia_aprendizagem.agenda("V1", dias, series, reamostragens=19, semente=3)
    cega = cega_aprendizagem.agenda("V1", dias, series, reamostragens=19, semente=3)
    sinal = datetime(2021, 1, 4, 12, tzinfo=UTC)
    intencoes = [
        Intencao(Campanha.DAY_TRADE_M15, regra, simbolo, Lado.COMPRA,
                 sinal, 10, sinal + timedelta(hours=1))
        for regra, simbolo in series
    ]
    a = referencia_aprendizagem.aplicar(intencoes, ref)
    b = cega_aprendizagem.aplicar(intencoes, cega)
    assert [(i.intencao_id, i.lado) for i in b] == [(i.intencao_id, i.lado) for i in a]
    assert [i.peso_admissao for i in b] == pytest.approx([i.peso_admissao for i in a], abs=1e-12)
