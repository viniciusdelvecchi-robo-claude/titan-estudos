from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

import titan.intencoes_compactas as compactas
from testes.test_f2_motor import _mundo_eventos_grande, params
from titan.intencoes_compactas import LoteIntencoesCompactas
from titan.motor import simular, simular_isoladas, simular_par
from titan.tipos_f2 import Campanha, Intencao, Lado


def _ns(instante: datetime) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = instante - epoch
    return (
        delta.days * 86_400_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1000
    )


def lote_de(intencoes, *, canonicalizar=True):
    intencoes = list(intencoes)
    simbolos = {i.simbolo for i in intencoes}
    campanhas = {i.campanha for i in intencoes}
    assert len(simbolos) == len(campanhas) == 1
    regras = tuple(dict.fromkeys(i.regra_id for i in intencoes))
    codigos = {r: k for k, r in enumerate(regras)}
    return LoteIntencoesCompactas.criar(
        next(iter(campanhas)), next(iter(simbolos)), regras,
        np.asarray([codigos[i.regra_id] for i in intencoes], dtype=np.uint16),
        np.asarray([1 if i.lado is Lado.COMPRA else -1 for i in intencoes], dtype=np.int8),
        np.asarray([_ns(i.barra_sinal_fim) for i in intencoes], dtype=np.int64),
        np.asarray([i.stop_pontos for i in intencoes], dtype=np.int64),
        np.asarray([_ns(i.hora_limite) for i in intencoes], dtype=np.int64),
        np.asarray([i.alvo_pontos or 0 for i in intencoes], dtype=np.int64),
        np.asarray([i.peso_admissao for i in intencoes], dtype=np.float64),
        canonicalizar=canonicalizar,
    )


def test_criar_canonicaliza_deduplica_e_preserva_id_campo_a_campo():
    t = datetime(2024, 3, 1, 10, tzinfo=UTC)
    a = Intencao(
        Campanha.DAY_TRADE_M15, "B", "X", Lado.VENDA, t, 12,
        t.replace(hour=12), 7, peso_admissao=2.0,
    )
    b = Intencao(
        Campanha.DAY_TRADE_M15, "A", "X", Lado.COMPRA, t, 10,
        t.replace(hour=13), None, peso_admissao=float("inf"),
    )
    lote = lote_de([a, b, a])
    esperado = sorted({i.intencao_id: i for i in (a, b)}.values(), key=lambda i: (
        i.barra_sinal_fim, i.intencao_id,
    ))
    assert [lote.materializar(k) for k in range(len(lote))] == esperado
    assert [lote.intencao_id(k) for k in range(len(lote))] == [i.intencao_id for i in esperado]
    assert all(not x.flags.writeable for x in (
        lote.regra, lote.lado, lote.sinal_fim_ns, lote.stop_pontos,
        lote.hora_limite_ns, lote.alvo_pontos, lote.peso_admissao,
    ))


def test_selecionar_exige_indices_canonicos_e_mantem_identidade():
    t = datetime(2024, 3, 1, 10, tzinfo=UTC)
    originais = [
        Intencao(
            Campanha.DAY_TRADE_M15, f"R{k}", "X",
            Lado.COMPRA if k % 2 == 0 else Lado.VENDA,
            t.replace(minute=15 * k), 10 + k, t.replace(hour=13),
        )
        for k in range(3)
    ]
    lote = lote_de(originais)
    selecionado = lote.selecionar(np.asarray([0, 2], dtype=np.int64))
    assert [selecionado.materializar(k) for k in range(2)] == [
        lote.materializar(0), lote.materializar(2),
    ]
    with pytest.raises(ValueError):
        lote.selecionar(np.asarray([2, 1], dtype=np.int64))
    with pytest.raises(TypeError):
        lote.selecionar(np.asarray([0], dtype=np.int32))


def test_criar_recusa_dtype_implicito():
    vazio_i64 = np.asarray([], dtype=np.int64)
    with pytest.raises(TypeError):
        LoteIntencoesCompactas.criar(
            Campanha.DAY_TRADE_M15, "X", ("R",),
            np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int8),
            vazio_i64, vazio_i64, vazio_i64, vazio_i64,
            np.asarray([], dtype=np.float64),
        )


def test_criar_canonico_confiavel_preserva_validacao_sem_recalcular_hash(monkeypatch):
    t = datetime(2024, 3, 1, 10, tzinfo=UTC)
    base = lote_de([
        Intencao(
            Campanha.DAY_TRADE_M15, "R", "X", Lado.COMPRA,
            t, 10, t.replace(hour=13),
        ),
    ])
    monkeypatch.setattr(
        compactas, "_digest_compacto",
        lambda *args: (_ for _ in ()).throw(AssertionError("rehash")),
    )
    copia = LoteIntencoesCompactas.criar(
        base.campanha, base.simbolo, base.regra_ids,
        *(np.ascontiguousarray(x) for x in (
            base.regra, base.lado, base.sinal_fim_ns, base.stop_pontos,
            base.hora_limite_ns, base.alvo_pontos, base.peso_admissao,
        )), canonicalizar=False,
    )
    assert len(copia) == 1
    with pytest.raises(ValueError):
        LoteIntencoesCompactas.criar(
            base.campanha, base.simbolo, base.regra_ids,
            np.ascontiguousarray(base.regra), np.array([0], dtype=np.int8),
            np.ascontiguousarray(base.sinal_fim_ns),
            np.ascontiguousarray(base.stop_pontos),
            np.ascontiguousarray(base.hora_limite_ns),
            np.ascontiguousarray(base.alvo_pontos),
            np.ascontiguousarray(base.peso_admissao), canonicalizar=False,
        )


@pytest.mark.parametrize("pessimista", [False, True])
def test_lote_e_objetos_iguais_nas_isoladas(pessimista):
    por_chave, dados, contratos, calendario = _mundo_eventos_grande(denso=True)
    objetos = {k: v for k, v in por_chave.items() if k[0] == "S0" and k[1] < 4}
    lotes = {k: lote_de(v) for k, v in objetos.items()}
    p = params(
        capital_constante=True, risco_por_operacao=0.005,
        pessimista=pessimista,
    )
    esperado = simular_isoladas(
        objetos, dados, contratos, calendario, p, somente_series=True,
    )
    atual = simular_isoladas(
        lotes, dados, contratos, calendario, p, somente_series=True,
    )
    assert atual == esperado


def test_lotes_e_objetos_iguais_no_par_campo_a_campo():
    por_chave, dados, contratos, calendario = _mundo_eventos_grande(denso=True)
    por_ativo = {}
    for simbolo in ("S0", "S1"):
        por_ativo[simbolo] = [
            i for (s, regra), itens in por_chave.items()
            if s == simbolo and regra < 4 for i in itens
        ]
    objetos = [i for s in sorted(por_ativo) for i in por_ativo[s]]
    lotes = [lote_de(por_ativo[s]) for s in sorted(por_ativo)]
    normal = params(
        capital_constante=True, risco_por_operacao=0.005,
        pessimista=False,
    )
    pessimista = replace(normal, pessimista=True)
    referencia = tuple(
        simular(
            objetos, dados, contratos, calendario, p,
            somente_series=True, _otimizar=False,
        )
        for p in (normal, pessimista)
    )
    esperado = simular_par(
        objetos, dados, contratos, calendario, normal, pessimista,
        somente_series=True,
    )
    atual = simular_par(
        lotes, dados, contratos, calendario, normal, pessimista,
        somente_series=True,
    )
    assert esperado == referencia
    assert atual == referencia
