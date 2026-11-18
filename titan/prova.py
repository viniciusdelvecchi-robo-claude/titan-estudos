"""Prova de referência do pré-registro 8 (seção 3 do plano; F2-1, F2-8; protocolo em docs/protocolo_prova_c1.md).

H0: retorno líquido esperado da série de prova <= 0 (emenda C2). Unilateral, sobre a média, por dois
métodos que precisam passar os dois, nas séries normal e pessimista: t de Newey-West e bootstrap
estacionário estudentizado (Politis e Romano, 1994; bloco por Politis e White, 2004, com a correção de
Patton, Politis e White, 2009). Dois candidatos em ordem de Holm, com a folga (ParamsPortao.niveis()).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from titan.tipos_f2 import (EntradaProva, ParamsPortao, ResultadoMotor, VereditoCandidato,
                            VereditoPortao)

SHARPE_SUSPEITO = 3.0  # T13: Sharpe anual acima disto fica retido até a auditoria de vazamento
DIAS_ANO = 260.3  # pregões por ano medidos no servidor (micro-barra de domingo; plano v18)


def _phi(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def lag_newey_west(n: int) -> int:
    """Janela automática de Newey e West (1994), regra de bolso: piso de 4·(n/100)^(2/9)."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _nw_var_media(x: np.ndarray, lag: int) -> np.ndarray:
    """Variância de Newey-West da média; aceita matriz (linhas = séries)."""
    x = np.atleast_2d(x)
    n = x.shape[1]
    d = x - x.mean(axis=1, keepdims=True)
    s = (d * d).sum(axis=1) / n
    for l in range(1, lag + 1):
        s = s + 2.0 * (1.0 - l / (lag + 1.0)) * (d[:, l:] * d[:, :-l]).sum(axis=1) / n
    return np.maximum(s, 0.0) / n


def t_newey_west(x: np.ndarray) -> tuple[float, float]:
    """(t, p unilateral H1: média > 0) com aproximação normal."""
    x = np.asarray(x, dtype=float)
    v = float(_nw_var_media(x, lag_newey_west(len(x)))[0])
    if v <= 0:
        return (math.inf if x.mean() > 0 else -math.inf), (0.0 if x.mean() > 0 else 1.0)
    t = float(x.mean() / math.sqrt(v))
    return t, 1.0 - _phi(t)


def bloco_politis_white(x: np.ndarray) -> float:
    """Tamanho médio de bloco para o bootstrap estacionário (Politis-White 2004 / PPW 2009)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    g0 = float(d @ d) / n
    if g0 == 0:
        return 1.0
    kn = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    mmax = int(math.ceil(math.sqrt(n))) + kn
    bmax = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    c = 2.0 * math.sqrt(math.log10(n) / n)
    rho = np.array([float(d[k:] @ d[:-k]) / n / g0 for k in range(1, mmax + kn + 1)])
    mhat = mmax
    for m in range(1, mmax + 1):
        if np.all(np.abs(rho[m: m + kn]) < c):
            mhat = m
            break
    M = min(2 * mhat, mmax)
    ks = np.arange(-M, M + 1)
    lam = np.where(np.abs(ks / M) <= 0.5, 1.0, np.where(np.abs(ks / M) <= 1.0, 2.0 * (1.0 - np.abs(ks / M)), 0.0))
    R = np.array([g0 if k == 0 else float(d[abs(k):] @ d[:-abs(k)]) / n for k in ks])
    G = float((lam * np.abs(ks) * R).sum())
    g = float((lam * R).sum())
    if g <= 0:
        return 1.0
    b = (2.0 * G * G / (2.0 * g * g)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 1.0), bmax))


def indices_estacionarios(n: int, bloco: float, reamostragens: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano: começo uniforme; a cada passo, novo começo com prob. 1/bloco; circular."""
    p = 1.0 / bloco
    novo = rng.random((reamostragens, n)) < p
    novo[:, 0] = True
    inicios = rng.integers(0, n, size=(reamostragens, n))
    idx = np.empty((reamostragens, n), dtype=np.int64)
    idx[:, 0] = inicios[:, 0]
    for j in range(1, n):
        idx[:, j] = np.where(novo[:, j], inicios[:, j], (idx[:, j - 1] + 1) % n)
    return idx


def p_bootstrap(x: np.ndarray, reamostragens: int, semente: int) -> float:
    """Bootstrap estacionário estudentizado: p = (1 + #{t* >= t}) / (B + 1), t* = (m* - m)/se*."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    lag = lag_newey_west(n)
    v = float(_nw_var_media(x, lag)[0])
    if v <= 0:
        return 0.0 if x.mean() > 0 else 1.0
    t = x.mean() / math.sqrt(v)
    rng = np.random.default_rng(semente)
    b = bloco_politis_white(x)
    contagem = 0
    feitos = 0
    lote = max(1, min(reamostragens, 2_000_000 // max(n, 1)))
    while feitos < reamostragens:
        k = min(lote, reamostragens - feitos)
        amostras = x[indices_estacionarios(n, b, k, rng)]
        se = np.sqrt(_nw_var_media(amostras, lag))
        tst = np.where(se > 0, (amostras.mean(axis=1) - x.mean()) / np.where(se > 0, se, 1.0), 0.0)
        contagem += int((tst >= t).sum())
        feitos += k
    return (1 + contagem) / (reamostragens + 1)


@dataclass(frozen=True)
class _Corte:
    retorno: np.ndarray
    blocos: int


def _janela(r: ResultadoMotor, inicio, fim=None) -> _Corte:
    datas = np.array(r.datas, dtype="datetime64[D]")
    m = datas >= np.datetime64(inicio)
    if fim is not None:
        m &= datas < np.datetime64(fim)
    # A1: o bloco é a data (dez primeiros caracteres do rótulo); apostas do mesmo dia = um bloco
    blocos_validos = {b[:10] for b in r.blocos if b and np.datetime64(b[:10]) >= np.datetime64(inicio)
                      and (fim is None or np.datetime64(b[:10]) < np.datetime64(fim))}
    return _Corte(np.asarray(r.retorno, dtype=float)[m], len(blocos_validos))


def avaliar(e: EntradaProva, p: ParamsPortao) -> dict:
    """Estatísticas de um candidato, sem decidir o nível (Holm decide depois)."""
    prova_n = _janela(e.normal, e.janela_inicio, e.cofre_inicio)
    prova_p = _janela(e.pessimista, e.janela_inicio, e.cofre_inicio)
    cofre = _janela(e.normal, e.cofre_inicio)
    _, p_nw = t_newey_west(prova_n.retorno)
    _, p_nw_p = t_newey_west(prova_p.retorno)
    p_bs = p_bootstrap(prova_n.retorno, p.reamostragens, p.semente)
    p_bs_p = p_bootstrap(prova_p.retorno, p.reamostragens, p.semente + 1)
    if len(cofre.retorno) >= 2:
        t_cofre, _ = t_newey_west(cofre.retorno)
        p_cofre_neg = _phi(t_cofre)  # p unilateral de H1: média do Cofre < 0
    else:
        p_cofre_neg = 1.0
    partidas = list(e.partidas_sentinela)
    frac_rev = (sum(partidas) / len(partidas)) if partidas else 1.0
    x = prova_n.retorno
    sharpe = float(x.mean() / x.std(ddof=1) * math.sqrt(DIAS_ANO)) if len(x) > 1 and x.std(ddof=1) > 0 else 0.0
    return dict(p=max(p_nw, p_bs, p_nw_p, p_bs_p), p_nw=p_nw, p_bs=p_bs, p_nw_p=p_nw_p, p_bs_p=p_bs_p,
                blocos=prova_n.blocos, p_cofre_neg=p_cofre_neg, frac_rev=frac_rev, sharpe=sharpe,
                media=float(x.mean()) if len(x) else 0.0)


def testar(candidatos: list[EntradaProva], params: ParamsPortao) -> VereditoPortao:
    if len(candidatos) != params.passos_holm:
        raise ValueError("número de candidatos difere dos passos de Holm declarados")
    est = {c.candidato: avaliar(c, params) for c in candidatos}
    ordem = tuple(sorted(est, key=lambda k: (est[k]["p"], k)))
    niveis = params.niveis()
    vereditos = []
    seguir = True
    for passo, nome in enumerate(ordem):
        s = est[nome]
        motivos = []
        nivel = niveis[passo] if seguir else None
        if nivel is None:
            motivos.append("holm_parou")
        elif s["p"] > nivel:
            motivos.append("p_acima_do_nivel")
        if s["blocos"] < params.apostas_min:
            motivos.append("blocos_insuficientes")
        if s["p_cofre_neg"] < params.nivel_cofre:
            motivos.append("veto_do_cofre")
        if s["frac_rev"] > params.revogacao_max:
            motivos.append("sentinela_revogaria")
        if s["sharpe"] > SHARPE_SUSPEITO:
            motivos.append("retido_T13")
        aprovado = not motivos
        seguir = seguir and aprovado  # conservador: o passo seguinte só abre com o anterior aprovado
        vereditos.append(VereditoCandidato(
            candidato=nome, p_newey_west=s["p_nw"], p_bootstrap=s["p_bs"], p_newey_west_pessimista=s["p_nw_p"],
            p_bootstrap_pessimista=s["p_bs_p"], blocos_independentes=s["blocos"], p_cofre_negativo=s["p_cofre_neg"],
            fracao_revogada=s["frac_rev"], nivel_aplicado=nivel, aprovado=aprovado, motivos=tuple(motivos)))
    diag = {}
    for c in candidatos:  # emenda C2: diagnóstico contra a exposição à deriva; não decide
        r = c.normal
        diag[c.candidato] = dict(media=est[c.candidato]["media"],
                                 exposicao_comprada_media=float(np.mean(r.exposicao_comprada)) if r.datas else 0.0,
                                 exposicao_vendida_media=float(np.mean(r.exposicao_vendida)) if r.datas else 0.0,
                                 sharpe_anual=est[c.candidato]["sharpe"])
    return VereditoPortao(ordem=ordem, candidatos=tuple(vereditos), diagnostico_deriva=diag)
