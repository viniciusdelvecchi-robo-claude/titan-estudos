"""Reimplementacao cega da prova da campanha 1, somente por A1--A11.

Este modulo nao depende de ``titan.prova``. A referencia e usada apenas como
caixa-preta pelo teste cruzado F2-7.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import ceil, erfc, floor, inf, log10, sqrt
from typing import Mapping, Sequence

import numpy as np

from titan.tipos_f2 import (
    EntradaProva,
    ParamsPortao,
    ResultadoMotor,
    VereditoCandidato,
    VereditoPortao,
)


RAIZ_2 = sqrt(2.0)
DIAS_ANO = 260.3


def _sf_normal(t: float) -> float:
    return 0.5 * erfc(t / RAIZ_2)


def _cdf_normal(t: float) -> float:
    return 0.5 * erfc(-t / RAIZ_2)


def _lag_nw(n: int) -> int:
    return floor(4.0 * (n / 100.0) ** (2.0 / 9.0))


def _gammas(x: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(x)
    d = x - float(np.mean(x))
    saida = np.zeros(max_lag + 1, dtype=np.float64)
    for lag in range(min(max_lag, n - 1) + 1):
        saida[lag] = float(np.dot(d[lag:], d[: n - lag])) / n
    return saida


def _nw(x: np.ndarray, lag: int | None = None) -> tuple[float, float, float]:
    """Retorna t, p unilateral superior e variancia da media (A4)."""
    n = len(x)
    if n == 0:
        raise ValueError("serie de prova vazia")
    media = float(np.mean(x))
    lag = _lag_nw(n) if lag is None else lag
    gamma = _gammas(x, lag)
    s = gamma[0]
    for k in range(1, lag + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gamma[k]
    variancia = max(float(s), 0.0) / n
    if variancia == 0.0:
        return (inf if media > 0 else 0.0), (0.0 if media > 0 else 1.0), 0.0
    t = media / sqrt(variancia)
    return t, _sf_normal(t), variancia


def _bloco_pw(x: np.ndarray) -> float:
    """Tamanho de bloco de Politis--White com correcao PPW (A5)."""
    n = len(x)
    if n == 0:
        raise ValueError("serie de prova vazia")
    k_janela = max(5, ceil(sqrt(log10(n))))
    m_max = ceil(sqrt(n)) + k_janela
    b_max = ceil(min(3.0 * sqrt(n), n / 3.0))
    c = 2.0 * sqrt(log10(n) / n)
    gamma = _gammas(x, m_max + k_janela)
    if gamma[0] == 0.0:
        return 1.0
    rho = gamma / gamma[0]
    m_estimado = m_max
    for m in range(1, m_max + 1):
        if np.all(np.abs(rho[m + 1 : m + k_janela + 1]) < c):
            m_estimado = m
            break
    m_trapezio = min(2 * m_estimado, m_max)
    g = gamma[0]
    G = 0.0
    for k in range(1, m_trapezio + 1):
        s = k / m_trapezio
        lam = 1.0 if s <= 0.5 else 2.0 * (1.0 - s)
        g += 2.0 * lam * gamma[k]
        G += 2.0 * lam * k * gamma[k]
    if g <= 0.0:
        return 1.0
    bloco = ((G * G) / (g * g)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return min(max(bloco, 1.0), float(b_max))


def _variancias_nw_lote(amostras: np.ndarray, lag: int) -> np.ndarray:
    """A4 por linha, preservando a aritmetica e o denominador n."""
    n = amostras.shape[1]
    medias = np.mean(amostras, axis=1)
    d = amostras - medias[:, None]
    s = np.sum(d * d, axis=1) / n
    for k in range(1, lag + 1):
        gamma = np.sum(d[:, k:] * d[:, : n - k], axis=1) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    return np.maximum(s, 0.0) / n


def _bootstrap(x: np.ndarray, B: int, semente: int) -> float:
    n = len(x)
    lag = _lag_nw(n)
    t_original, _, _ = _nw(x, lag)
    media = float(np.mean(x))
    bloco = _bloco_pw(x)
    prob_reinicio = 1.0 / bloco
    rng = np.random.default_rng(semente)
    lote_max = max(1, floor(2_000_000 / n))
    extremos = 0
    feitos = 0
    while feitos < B:
        tamanho = min(B - feitos, lote_max)
        U = rng.random((tamanho, n))
        I = rng.integers(0, n, size=(tamanho, n))
        indices = np.empty((tamanho, n), dtype=np.int64)
        indices[:, 0] = I[:, 0]
        for j in range(1, n):
            indices[:, j] = np.where(
                U[:, j] < prob_reinicio,
                I[:, j],
                (indices[:, j - 1] + 1) % n,
            )
        amostras = x[indices]
        medias = np.mean(amostras, axis=1)
        variancias = _variancias_nw_lote(amostras, lag)
        t_estrela = np.zeros(tamanho, dtype=np.float64)
        positivas = variancias > 0.0
        t_estrela[positivas] = (medias[positivas] - media) / np.sqrt(
            variancias[positivas]
        )
        extremos += int(np.count_nonzero(t_estrela >= t_original))
        feitos += tamanho
    return (1.0 + extremos) / (B + 1.0)


def _trecho(resultado: ResultadoMotor, inicio: date, fim: date | None) -> np.ndarray:
    return np.asarray(
        [
            r
            for dia, r in zip(resultado.datas, resultado.retorno, strict=True)
            if dia >= inicio and (fim is None or dia < fim)
        ],
        dtype=np.float64,
    )


def _exposicao_prova(resultado: ResultadoMotor, inicio: date, fim: date) -> tuple[float, float]:
    comprada = [
        x for d, x in zip(resultado.datas, resultado.exposicao_comprada, strict=True)
        if inicio <= d < fim
    ]
    vendida = [
        x for d, x in zip(resultado.datas, resultado.exposicao_vendida, strict=True)
        if inicio <= d < fim
    ]
    return (
        float(np.mean(comprada)) if comprada else 0.0,
        float(np.mean(vendida)) if vendida else 0.0,
    )


def _sharpe(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    desvio = float(np.std(x, ddof=1))
    media = float(np.mean(x))
    if desvio == 0.0:
        return inf if media > 0.0 else 0.0
    return media / desvio * sqrt(DIAS_ANO)


def _contar_blocos(entrada: EntradaProva) -> int:
    dias: set[date] = set()
    for rotulo in entrada.normal.blocos:
        try:
            dia = date.fromisoformat(rotulo[:10])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rotulo de bloco sem data ISO: {rotulo!r}") from exc
        if entrada.janela_inicio <= dia < entrada.cofre_inicio:
            dias.add(dia)
    return len(dias)


@dataclass(frozen=True, slots=True)
class _Calculado:
    entrada: EntradaProva
    p_nw: float
    p_boot: float
    p_nw_pess: float
    p_boot_pess: float
    p_candidato: float
    blocos: int
    p_cofre_negativo: float
    fracao_revogada: float
    sharpe: float
    media: float
    exposicao_comprada: float
    exposicao_vendida: float


def _calcular(entrada: EntradaProva, params: ParamsPortao) -> _Calculado:
    normal = _trecho(entrada.normal, entrada.janela_inicio, entrada.cofre_inicio)
    pessimista = _trecho(entrada.pessimista, entrada.janela_inicio, entrada.cofre_inicio)
    _, p_nw, _ = _nw(normal)
    _, p_nw_pess, _ = _nw(pessimista)
    p_boot = _bootstrap(normal, params.reamostragens, params.semente)
    p_boot_pess = _bootstrap(pessimista, params.reamostragens, params.semente + 1)
    cofre = _trecho(entrada.normal, entrada.cofre_inicio, None)
    if len(cofre) >= 2:
        t_cofre, _, _ = _nw(cofre)
        p_cofre = _cdf_normal(t_cofre)
    else:
        p_cofre = 1.0
    partidas = tuple(bool(x) for x in entrada.partidas_sentinela)
    fracao = sum(partidas) / len(partidas) if partidas else 1.0
    exp_c, exp_v = _exposicao_prova(
        entrada.normal, entrada.janela_inicio, entrada.cofre_inicio
    )
    return _Calculado(
        entrada, p_nw, p_boot, p_nw_pess, p_boot_pess,
        max(p_nw, p_boot, p_nw_pess, p_boot_pess),
        _contar_blocos(entrada), p_cofre, fracao, _sharpe(normal),
        float(np.mean(normal)), exp_c, exp_v,
    )


def testar(
    candidatos: Sequence[EntradaProva], params: ParamsPortao
) -> VereditoPortao:
    """Executa A1--A10 e devolve os contratos comuns de ``tipos_f2``."""
    if len(candidatos) != params.passos_holm:
        raise ValueError("numero de candidatos deve ser igual a passos_holm")
    nomes = [c.candidato for c in candidatos]
    if len(set(nomes)) != len(nomes):
        raise ValueError("nomes de candidato duplicados")
    calculados = sorted(
        (_calcular(c, params) for c in candidatos),
        key=lambda x: (x.p_candidato, x.entrada.candidato),
    )
    vereditos: list[VereditoCandidato] = []
    holm_aberto = True
    for passo, c in enumerate(calculados):
        nivel: float | None = params.niveis()[passo] if holm_aberto else None
        motivos: list[str] = []
        if nivel is None:
            motivos.append("holm_parou")
        else:
            if c.p_candidato > nivel:
                motivos.append("p_acima_do_nivel")
            if c.blocos < params.apostas_min:
                motivos.append("blocos_insuficientes")
            if c.p_cofre_negativo < params.nivel_cofre:
                motivos.append("veto_do_cofre")
            if c.fracao_revogada > params.revogacao_max:
                motivos.append("sentinela_revogaria")
            if c.sharpe > 3.0:
                motivos.append("retido_T13")
        aprovado = not motivos
        vereditos.append(VereditoCandidato(
            candidato=c.entrada.candidato,
            p_newey_west=c.p_nw,
            p_bootstrap=c.p_boot,
            p_newey_west_pessimista=c.p_nw_pess,
            p_bootstrap_pessimista=c.p_boot_pess,
            blocos_independentes=c.blocos,
            p_cofre_negativo=c.p_cofre_negativo,
            fracao_revogada=c.fracao_revogada,
            nivel_aplicado=nivel,
            aprovado=aprovado,
            motivos=tuple(motivos),
        ))
        if not aprovado:
            holm_aberto = False
    diagnostico: Mapping[str, object] = {
        c.entrada.candidato: {
            "media_prova": c.media,
            "exposicao_comprada_media": c.exposicao_comprada,
            "exposicao_vendida_media": c.exposicao_vendida,
            "sharpe_anual": c.sharpe,
        }
        for c in calculados
    }
    return VereditoPortao(
        ordem=tuple(c.entrada.candidato for c in calculados),
        candidatos=tuple(vereditos),
        diagnostico_deriva=diagnostico,
    )


__all__ = ["testar"]
