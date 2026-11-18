"""Parte C do protocolo C1, reimplementada sem consultar a referência."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
import math
from statistics import median
from typing import Mapping, Sequence

import numpy as np

from titan.calendario import offset_servidor_min
from titan.prova_cega_codex import _bloco_pw, _nw
from titan.tipos_f2 import Intencao

Chave = tuple[str, str]


@dataclass(frozen=True)
class Periodo:
    inicio: date
    fim: date | None
    pesos: Mapping[Chave, float] | None


def datas_reajuste(dias: Sequence[date]) -> list[date]:
    grade = sorted(set(dias))
    por_ano: dict[int, list[date]] = {}
    for d in grade:
        por_ano.setdefault(d.year, []).append(d)
    out = []
    for ano in sorted(por_ano):
        anterior = por_ano.get(ano - 1, [])
        if anterior and anterior[0] <= date(ano - 1, 1, 7):
            out.append(por_ano[ano][0])
    return out


def _elegiveis(series: Mapping[Chave, np.ndarray], corte: int) -> list[Chave]:
    return sorted(k for k, x in series.items() if np.count_nonzero(np.asarray(x[:corte])) >= 2)


def _t(x: np.ndarray) -> float:
    t, _, _ = _nw(np.asarray(x, dtype=float))
    return float(t)


def _indices_estacionarios(n: int, bloco: float, B: int, rng: np.random.Generator) -> np.ndarray:
    p = min(1.0, 1.0 / max(1.0, bloco))
    out = np.empty((B, n), dtype=np.int64)
    out[:, 0] = rng.integers(0, n, B)
    for j in range(1, n):
        reinicia = rng.random(B) < p
        out[:, j] = np.where(reinicia, rng.integers(0, n, B), (out[:, j - 1] + 1) % n)
    return out


def sobreviventes_v3(
    passado: Mapping[Chave, np.ndarray], reamostragens: int, semente: int
) -> set[Chave]:
    chaves = sorted(passado)
    if not chaves:
        return set()
    n = len(np.asarray(passado[chaves[0]]))
    blocos = [float(_bloco_pw(np.asarray(passado[k], dtype=float))) for k in chaves]
    bloco = median(blocos)
    idx = _indices_estacionarios(n, bloco, reamostragens, np.random.default_rng(semente))
    obs = {k: _t(np.asarray(passado[k], dtype=float)) for k in chaves}
    estrelas: dict[Chave, np.ndarray] = {}
    for k in chaves:
        x = np.asarray(passado[k], dtype=float)
        centrada = x - x.mean()
        estrelas[k] = np.asarray([_t(centrada[ii]) for ii in idx], dtype=float)
    restantes = set(chaves)
    rejeitadas: set[Chave] = set()
    while restantes:
        maximos = np.max(np.vstack([estrelas[k] for k in sorted(restantes)]), axis=0)
        critico = float(np.quantile(maximos, 0.90, method="higher"))
        novas = {k for k in restantes if obs[k] > critico}
        if not novas:
            break
        rejeitadas |= novas
        restantes -= novas
    return rejeitadas


def _pesos(
    variante: str,
    passado: Mapping[Chave, np.ndarray],
    reamostragens: int,
    semente: int,
) -> dict[Chave, float]:
    todas = sorted(passado)
    eleg = [k for k in todas if np.count_nonzero(passado[k]) >= 2]
    if variante == "V0":
        return {k: 1.0 for k in todas}
    ts = {k: _t(np.asarray(passado[k], dtype=float)) for k in eleg}
    if variante == "V1":
        return {k: max(0.0, ts.get(k, 0.0)) for k in todas}
    if variante in {"V2-20", "V2-50", "V2_20", "V2_50"}:
        N = int(variante.replace("-", "_").split("_")[1])
        escolhidas = set(sorted(eleg, key=lambda k: (-ts[k], k))[:N])
        return {k: 1.0 if k in escolhidas else 0.0 for k in todas}
    if variante == "V3":
        escolhidas = sobreviventes_v3({k: passado[k] for k in eleg}, reamostragens, semente)
        return {k: 1.0 if k in escolhidas else 0.0 for k in todas}
    raise ValueError(f"variante desconhecida: {variante}")


def agenda(
    variante: str,
    dias: Sequence[date],
    series: Mapping[Chave, np.ndarray],
    *,
    reamostragens: int,
    semente: int,
) -> list[Periodo]:
    grade = sorted(set(dias))
    if not grade:
        return []
    if any(len(x) != len(grade) for x in series.values()):
        raise ValueError("séries devem estar alinhadas na grade comum")
    marcos = datas_reajuste(grade)
    periodos: list[Periodo] = []
    primeiro_fim = marcos[0] if marcos else None
    periodos.append(Periodo(grade[0], primeiro_fim, None))
    for j, d in enumerate(marcos):
        corte = grade.index(d)
        passado = {k: np.asarray(x[:corte], dtype=float) for k, x in series.items()}
        pesos = _pesos(variante, passado, reamostragens, semente + d.year)
        periodos.append(Periodo(d, marcos[j + 1] if j + 1 < len(marcos) else None, pesos))
    return periodos


def aplicar(intencoes: Sequence[Intencao], periodos: Sequence[Periodo]) -> list[Intencao]:
    saida = []
    for i in intencoes:
        deslocamento = offset_servidor_min(i.barra_sinal_fim)
        dia = (i.barra_sinal_fim + timedelta(minutes=deslocamento)).date()
        p = next((p for p in periodos if p.inicio <= dia and (p.fim is None or dia < p.fim)), None)
        if p is None:
            continue
        peso = 1.0 if p.pesos is None else float(p.pesos.get((i.regra_id, i.simbolo), 0.0))
        if peso > 0:
            saida.append(replace(i, peso_admissao=peso))
    return saida


__all__ = ["Periodo", "agenda", "aplicar", "datas_reajuste", "sobreviventes_v3"]
