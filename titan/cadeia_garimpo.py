"""Cadeia do Garimpo num mundo sintético (F2-12 e F2-9): gramática → motor por regra×ativo → aprendizagem
→ motor do Garimpo (normal e pessimista) → estatísticas da prova (parte A). Nunca preço real.

O estudo de poder só escolhe a configuração do Garimpo; a sentinela e o Núcleo entram no F2-9/F2-10.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence

import numpy as np

from titan.aprendizagem import agenda, aplicar, aplicar_compacto, datas_reajuste
from titan.gramatica import gerar, gerar_compacto
from titan.motor import simular, simular_isoladas, simular_par
from titan.prova import avaliar
from titan.sintetico import Mundo
from titan.tipos_f2 import EntradaProva, Intencao, ParamsMotor, ParamsPortao, ResultadoMotor

Chave = tuple[str, str]


def params_motor(risco: float, pessimista: bool) -> ParamsMotor:
    # risco_aberto_alvo e limiar_m provisórios até o F2-10 fixar M (o estudo de poder não depende deles)
    return ParamsMotor(capital_usd=3000.0, risco_por_operacao=risco, risco_aberto_alvo=0.5, limiar_m=0.2,
                       fracao_um_terco=1 / 3, margem_max=0.5, limite_perda_dia=0.03, filtro_spread_mov=0.2,
                       filtro_janela_pregoes=60, pessimista=pessimista)


def intencoes(mundo: Mundo, compacto: bool = False) -> dict:
    """Por ativo: lista de Intencao (padrão) ou LoteIntencoesCompactas (compacto=True; mesmo conteúdo)."""
    if compacto:
        return {s: gerar_compacto(s, mundo.m15[s], mundo.sessoes[s]) for s in sorted(mundo.m15)}
    return {s: gerar(s, mundo.m15[s], mundo.sessoes[s]) for s in sorted(mundo.m15)}


def _e_compacto(its: Mapping) -> bool:
    from titan.intencoes_compactas import LoteIntencoesCompactas
    return any(isinstance(v, LoteIntencoesCompactas) for v in its.values())


def _rodar(its: Sequence[Intencao], mundo: Mundo, params: ParamsMotor, enxuto: bool = False) -> ResultadoMotor:
    simbolos = {i.simbolo for i in its}
    carregador = {s: mundo.barras_motor(s) for s in simbolos}
    r = simular(list(its), carregador, mundo.specs, mundo.calendario, params, somente_series=enxuto)
    if enxuto:  # a prova só usa as séries; livro, recusas, estados e conferências saem da memória
        r = ResultadoMotor(r.datas, r.pnl_usd, r.retorno, r.exposicao_comprada, r.exposicao_vendida, r.blocos,
                           (), (), (), ())
    return r


def _na_grade(r: ResultadoMotor, dias: Sequence[date]) -> np.ndarray:
    pos = {d: i for i, d in enumerate(dias)}
    x = np.zeros(len(dias))
    for d, v in zip(r.datas, r.retorno):
        if d in pos:
            x[pos[d]] += v
    return x


def series_isoladas(mundo: Mundo, its: Mapping[str, list[Intencao]], risco: float) -> dict[Chave, np.ndarray]:
    """C0: série de cada regra×ativo, motor rodando só com as intenções daquela chave (série normal)."""
    p = params_motor(risco, False)
    grupos: dict = {}
    if _e_compacto(its):
        for s, lote in its.items():
            for code in np.unique(lote.regra):
                grupos[(lote.regra_ids[int(code)], s)] = lote.selecionar(
                    np.flatnonzero(lote.regra == code).astype(np.int64))
    else:
        for lista in its.values():
            for i in lista:
                grupos.setdefault((i.regra_id, i.simbolo), []).append(i)
    carregador = {s: mundo.barras_motor(s) for s in {k[1] for k in grupos}}
    res = simular_isoladas(dict(sorted(grupos.items())), carregador, mundo.specs, mundo.calendario, p,
                            somente_series=True)
    return {k: _na_grade(res[k], mundo.dias) for k in sorted(grupos)}


def _so_ativo(mundo: Mundo, s: str) -> Mundo:
    return Mundo(mundo.dias, {s: mundo.m15[s]}, {s: mundo.sessoes[s]}, {s: mundo.specs[s]}, (), {})


def _preparar_ativo(args):
    """Num processo: gramática compacta e séries isoladas (todos os riscos) de UM ativo."""
    mini, s, riscos = args
    lote = gerar_compacto(s, mini.m15[s], mini.sessoes[s])
    return s, lote, {r: series_isoladas(mini, {s: lote}, r) for r in riscos}


def preparar(mundo: Mundo, riscos: Sequence[float], processos: int = 1) -> tuple[dict, dict]:
    """Gramática e séries isoladas por ativo em paralelo (são independentes entre ativos; resultado idêntico
    ao sequencial). Devolve (lotes por ativo, {risco: séries por chave})."""
    tarefas = [(_so_ativo(mundo, s), s, tuple(riscos)) for s in sorted(mundo.m15)]
    if processos > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(processos) as ex:
            partes = list(ex.map(_preparar_ativo, tarefas))
    else:
        partes = [_preparar_ativo(t) for t in tarefas]
    lotes = {s: lote for s, lote, _ in partes}
    series = {r: {k: v for _, _, por in partes for k, v in por[r].items()} for r in riscos}
    return lotes, {r: dict(sorted(series[r].items())) for r in riscos}


@dataclass(frozen=True)
class Janelas:
    janela_inicio: date
    cofre_inicio: date


def janelas_padrao(dias: Sequence[date], anos_cofre: float = 1.1) -> Janelas:
    """Prova começa no primeiro reajuste (o primeiro ano só aquece); Cofre = últimos `anos_cofre` anos."""
    marcos = datas_reajuste(list(dias))
    if not marcos:
        raise ValueError("mundo curto demais: sem reajuste")
    cofre = dias[max(0, len(dias) - int(round(anos_cofre * 260.3)))]
    return Janelas(marcos[0], cofre)


@dataclass(frozen=True)
class Avaliacao:
    variante: str
    risco: float
    p: float
    blocos: int
    p_cofre_neg: float
    sharpe: float
    media: float


def avaliar_configuracao(mundo: Mundo, its: Mapping[str, list[Intencao]], series: Mapping[Chave, np.ndarray],
                         variante: str, risco: float, janelas: Janelas, portao: ParamsPortao) -> Avaliacao:
    per = agenda(variante, mundo.dias, series, reamostragens=portao.reamostragens, semente=portao.semente)
    if _e_compacto(its):
        lotes = aplicar_compacto(its, per)
        todas = [lotes[s] for s in sorted(lotes) if len(lotes[s])]
        carregador = {lote.simbolo: mundo.barras_motor(lote.simbolo) for lote in todas}
    else:
        todas = aplicar([i for s in sorted(its) for i in its[s]], per)
        carregador = {s: mundo.barras_motor(s) for s in {i.simbolo for i in todas}}
    normal, pess = simular_par(todas, carregador, mundo.specs, mundo.calendario, params_motor(risco, False),
                               params_motor(risco, True), somente_series=True)  # um pré-cálculo, dois estados
    e = EntradaProva("Garimpo", normal, pess, janelas.janela_inicio, janelas.cofre_inicio, [False])
    s = avaliar(e, portao)
    return Avaliacao(variante, risco, s["p"], s["blocos"], s["p_cofre_neg"], s["sharpe"], s["media"])
