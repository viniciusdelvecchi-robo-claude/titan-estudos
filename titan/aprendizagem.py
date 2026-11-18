"""Aprendizagem do Garimpo, campanha 1 (parte C de docs/protocolo_prova_c1.md).

Entrada: as séries diárias de retorno de cada regra×ativo produzidas pelo motor isoladamente, alinhadas
numa grade comum de dias de servidor (0 nos dias sem posição). Saída: pesos por período anual, que
filtram (peso 0 não emite) e ordenam (Intencao.peso_admissao) as intenções. Nada aqui lê o futuro: o peso
do período que começa no dia D usa só dias < D.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from titan.calendario import offset_servidor_min
from titan.prova import _nw_var_media, bloco_politis_white, indices_estacionarios, lag_newey_west
from titan.tipos_f2 import Intencao

Chave = tuple[str, str]  # (regra_id, simbolo)

VARIANTES = ("V0", "V1", "V2_20", "V2_50", "V3")
RISCOS = (0.005, 0.01, 0.02)  # C3
CONFIGURACOES = tuple((v, r) for v in VARIANTES for r in RISCOS)  # 15 configurações conjuntas do F2-12
NIVEL_V3 = 0.10


def dia_servidor(instante_utc: datetime) -> date:
    return (instante_utc + timedelta(minutes=offset_servidor_min(instante_utc))).date()


def datas_reajuste(dias: Sequence[date]) -> list[date]:
    """C0: primeiro pregão em ou após 1º de janeiro, desde que haja 12 meses de passado na grade."""
    if not dias:
        return []
    primeiro = dias[0]
    out, ano_visto = [], None
    for d in dias:
        if d.year != ano_visto:
            ano_visto = d.year
            if d != primeiro and _doze_meses(primeiro, d):  # 1º pregão do ano, mesmo fora de janeiro
                out.append(d)
    return out


def _doze_meses(primeiro: date, d: date) -> bool:
    """12 meses = o ano civil anterior inteiro na grade: a grade começa antes dele ou na 1ª semana dele."""
    return primeiro.year < d.year - 1 or (primeiro.year == d.year - 1 and primeiro.month == 1
                                         and primeiro.day <= 7)


def t_por_serie(m: np.ndarray) -> np.ndarray:
    """t de Newey-West (A4) linha a linha; var 0 → +inf se média > 0, −inf senão."""
    m = np.atleast_2d(np.asarray(m, dtype=float))
    n = m.shape[1]
    v = _nw_var_media(m, lag_newey_west(n))
    med = m.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(v > 0, med / np.sqrt(np.where(v > 0, v, 1.0)), np.where(med > 0, np.inf, -np.inf))
    return t


def _elegiveis(m: np.ndarray) -> np.ndarray:
    """Série elegível = ao menos dois dias com retorno diferente de zero no passado."""
    return (m != 0).sum(axis=1) >= 2


def pesos_v1(chaves: Sequence[Chave], m: np.ndarray) -> dict[Chave, float]:
    t = t_por_serie(m)
    el = _elegiveis(m)
    return {k: (float(max(t[i], 0.0)) if el[i] else 0.0) for i, k in enumerate(chaves)}  # +inf permitido


def pesos_v2(chaves: Sequence[Chave], m: np.ndarray, n_top: int) -> dict[Chave, float]:
    t = t_por_serie(m)
    el = _elegiveis(m)
    cand = [i for i in range(len(chaves)) if el[i]]
    cand.sort(key=lambda i: (-t[i], chaves[i]))  # empate: (regra, ativo) em ordem alfabética
    escolhidos = set(cand[:n_top])
    return {k: (1.0 if i in escolhidos else 0.0) for i, k in enumerate(chaves)}


def sobreviventes_v3(chaves: Sequence[Chave], m: np.ndarray, reamostragens: int, semente: int,
                     nivel: float = NIVEL_V3) -> set[Chave]:
    """Passos de Romano-Wolf (2005) com t estudentizado e bootstrap estacionário conjunto (A6).

    Mesmos índices para todas as séries (preserva a dependência entre elas). Bloco = mediana dos blocos
    de Politis-White das séries elegíveis. Rejeita H0_j (média <= 0) se t_j > quantil (1 − nível) do máximo
    de t*_k sobre as hipóteses ainda não rejeitadas; repete até nenhuma nova rejeição.
    """
    el = np.flatnonzero(_elegiveis(m))
    if len(el) == 0:
        return set()
    x = m[el]
    n = x.shape[1]
    lag = lag_newey_west(n)
    t = t_por_serie(x)
    bloco = float(np.median([bloco_politis_white(s) for s in x]))
    rng = np.random.default_rng(semente)
    med = x.mean(axis=1)
    tst = np.empty((reamostragens, len(el)))
    lote = max(1, min(reamostragens, 2_000_000 // max(n, 1)))
    feitos = 0
    while feitos < reamostragens:
        k = min(lote, reamostragens - feitos)
        idx = indices_estacionarios(n, bloco, k, rng)
        for j in range(k):
            a = x[:, idx[j]]
            se = np.sqrt(_nw_var_media(a, lag))
            tst[feitos + j] = np.where(se > 0, (a.mean(axis=1) - med) / np.where(se > 0, se, 1.0), 0.0)
        feitos += k
    ativas = np.ones(len(el), dtype=bool)
    rejeitadas = np.zeros(len(el), dtype=bool)
    while ativas.any():
        maximo = tst[:, ativas].max(axis=1)
        c = float(np.quantile(maximo, 1.0 - nivel, method="higher"))
        novas = ativas & (t > c)
        if not novas.any():
            break
        rejeitadas |= novas
        ativas &= ~novas
    return {chaves[el[i]] for i in np.flatnonzero(rejeitadas)}


def pesos(variante: str, chaves: Sequence[Chave], m: np.ndarray, *, reamostragens: int, semente: int
          ) -> dict[Chave, float]:
    if variante == "V0":
        return {k: 1.0 for k in chaves}
    if variante == "V1":
        return pesos_v1(chaves, m)
    if variante in ("V2_20", "V2_50"):
        return pesos_v2(chaves, m, int(variante[3:]))
    if variante == "V3":
        sob = sobreviventes_v3(chaves, m, reamostragens, semente)
        return {k: (1.0 if k in sob else 0.0) for k in chaves}
    raise ValueError(f"variante desconhecida: {variante}")


@dataclass(frozen=True)
class Periodo:
    inicio: date  # inclusivo
    fim: date | None  # exclusivo
    pesos: Mapping[Chave, float] | None  # None = pesos iguais (antes do primeiro reajuste)


def agenda(variante: str, dias: Sequence[date], series: Mapping[Chave, np.ndarray], *, reamostragens: int,
           semente: int) -> list[Periodo]:
    """Pesos de cada período anual. `series[k][i]` é o retorno de k no dia `dias[i]`."""
    dias = list(dias)
    chaves = sorted(series)
    m = np.vstack([np.asarray(series[k], dtype=float) for k in chaves]) if chaves else np.zeros((0, len(dias)))
    if m.shape[1] != len(dias):
        raise ValueError("séries desalinhadas da grade de dias")
    marcos = datas_reajuste(dias)
    out = [Periodo(dias[0] if dias else date.min, marcos[0] if marcos else None, None)]
    for i, d in enumerate(marcos):
        corte = dias.index(d)  # só dias < d: até a véspera
        p = pesos(variante, chaves, m[:, :corte], reamostragens=reamostragens, semente=semente + d.year)
        out.append(Periodo(d, marcos[i + 1] if i + 1 < len(marcos) else None, p))
    return out


def aplicar_compacto(lotes: Mapping[str, Any], periodos: Sequence[Periodo]) -> dict[str, Any]:
    """C1 sobre LoteIntencoesCompactas: mesma regra de aplicar(), linha a linha, sem criar Intencao.

    Preserva a ordem relativa, remove peso 0 e troca só peso_admissao (fora da identidade)."""
    from titan.intencoes_compactas import LoteIntencoesCompactas
    inicios = [p.inicio for p in periodos]
    out = {}
    for simbolo, lote in lotes.items():
        n = len(lote)
        if n == 0:
            out[simbolo] = lote
            continue
        unicos, inv = np.unique(lote.sinal_fim_ns, return_inverse=True)
        k_unico = np.array([max(0, bisect_right(inicios, dia_servidor(datetime.fromtimestamp(int(u) // 10**9, UTC)
                                                                          + timedelta(microseconds=(int(u) % 10**9) // 1000))) - 1)
                            for u in unicos], dtype=np.int64)
        k = k_unico[inv]
        tabela = np.empty((len(periodos), len(lote.regra_ids)))
        for p, per in enumerate(periodos):
            for r, rid in enumerate(lote.regra_ids):
                tabela[p, r] = 1.0 if per.pesos is None else float(per.pesos.get((rid, simbolo), 0.0))
        w = tabela[k, lote.regra.astype(np.int64)]
        manter = np.flatnonzero(w > 0)
        out[simbolo] = LoteIntencoesCompactas.criar(
            lote.campanha, lote.simbolo, lote.regra_ids,
            *(np.ascontiguousarray(a[manter]) for a in (lote.regra, lote.lado, lote.sinal_fim_ns, lote.stop_pontos,
                                                        lote.hora_limite_ns, lote.alvo_pontos)),
            np.ascontiguousarray(w[manter], dtype=np.float64), canonicalizar=False)
    return out


def aplicar(intencoes: Iterable[Intencao], periodos: Sequence[Periodo]) -> list[Intencao]:
    """C1: peso 0 não emite; os demais saem com peso_admissao = peso do período do dia do sinal."""
    out = []
    inicios = [p.inicio for p in periodos]
    dia_cache: dict[datetime, int] = {}  # instante → índice do período (o dia de servidor custa fuso horário)
    for it in intencoes:
        k = dia_cache.get(it.barra_sinal_fim)
        if k is None:
            k = max(0, bisect_right(inicios, dia_servidor(it.barra_sinal_fim)) - 1)
            dia_cache[it.barra_sinal_fim] = k
        per = periodos[k]
        w = 1.0 if per.pesos is None else float(per.pesos.get((it.regra_id, it.simbolo), 0.0))
        if w > 0:
            out.append(it if w == it.peso_admissao else replace(it, peso_admissao=w))
    return out
