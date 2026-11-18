"""Parte B do protocolo C1, reimplementada sem consultar a referência."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
import math
from statistics import mean
from typing import Mapping, Sequence

import numpy as np

from titan.tipos import Sessao
from titan.tipos_f2 import Campanha, Intencao, Lado
from titan.visao import Barras, de_ns, ns

M15 = timedelta(minutes=15)
H1 = timedelta(hours=1)


def _media_desvio_exato(valores: Sequence[float]) -> tuple[float, float]:
    """B3/B4: redução binária reproduzível, exatamente como o texto congelado."""
    n = len(valores)
    if n == 0:
        return 0.0, 0.0
    m = math.fsum(valores) / n
    if n < 2:
        return m, 0.0
    s = math.sqrt(math.fsum((x - m) ** 2 for x in valores) / (n - 1))
    return m, s


def _h1(m15: Barras) -> Barras:
    linhas = []
    for i in range(0, len(m15), 4):
        j = min(i + 4, len(m15))
        if j - i != 4:
            continue
        ts = [de_ns(int(x)) for x in m15.ts[i:j]]
        if ts[0].minute != 0 or any(ts[k] - ts[k - 1] != M15 for k in range(1, 4)):
            continue
        linhas.append((m15.ts[i], m15.o[i], max(m15.h[i:j]), min(m15.l[i:j]), m15.c[j - 1], m15.spread[j - 1]))
    if not linhas:
        z = np.empty(0, np.int64)
        return Barras("H1", z, z, z, z, z, z)
    col = [np.asarray([x[k] for x in linhas], dtype=np.int64) for k in range(6)]
    return Barras("H1", *col)


def _por_sessao(b: Barras, sessoes: Mapping[date, Sessao]) -> dict[date, list[int]]:
    out: dict[date, list[int]] = defaultdict(list)
    for i, valor in enumerate(b.ts):
        t = de_ns(int(valor))
        for d, s in sessoes.items():
            if s.abertura_utc <= t < s.fechamento_utc:
                out[d].append(i)
                break
    return out


def _atr(b: Barras, fim: datetime, n: int = 20) -> int | None:
    k = int(np.searchsorted(b.fim, ns(fim), side="right"))
    if k < n:
        return None
    tr = []
    for i in range(k - n, k):
        anterior = int(b.c[i - 1]) if i else int(b.o[i])
        tr.append(max(int(b.h[i] - b.l[i]), abs(int(b.h[i]) - anterior), abs(int(b.l[i]) - anterior)))
    return max(1, math.ceil(3 * mean(tr)))


def _intencao(regra: str, simbolo: str, lado: Lado, sinal: datetime, stop: int, limite: datetime) -> Intencao | None:
    if sinal >= limite:
        return None
    return Intencao(Campanha.DAY_TRADE_M15, regra, simbolo, lado, sinal, stop, limite)


def gerar(simbolo: str, m15: Barras, sessoes: Mapping[date, Sessao]) -> list[Intencao]:
    """Gera as 32 regras da referência de sessão da campanha 1.

    B6 e a segunda referência dos índices exigem o calendário de bolsa por
    símbolo, que não faz parte da assinatura textual ``gerar``; a comparação
    cega usa ativo sintético não índice, para o qual o total declarado é 32.
    """
    if m15.tempo_grafico != "M15":
        raise ValueError("parte B exige M15")
    h1 = _h1(m15)
    por_tf = {"M15": (m15, M15), "H1": (h1, H1)}
    grupos = {tf: _por_sessao(b, sessoes) for tf, (b, _) in por_tf.items()}
    saida: list[Intencao] = []
    dias = sorted(sessoes)
    fatias30: dict[tuple[date, int], float] = {}
    fechamento_por_fim = {
        int(fim): int(fechamento)
        for fim, fechamento in zip(m15.fim, m15.c, strict=True)
    }
    for d in dias:
        s = sessoes[d]
        for minuto in range(0, 24 * 60, 30):
            a = s.abertura_utc + timedelta(minutes=minuto)
            z = a + timedelta(minutes=30)
            c0 = fechamento_por_fim.get(ns(a))
            c1 = fechamento_por_fim.get(ns(z))
            if c0 is not None and c1 is not None:
                fatias30[(d, minuto)] = c1 / c0 - 1.0

    # B1 e B2, referência da sessão.
    for di, d in enumerate(dias):
        s = sessoes[d]
        if s.feriado or s.abertura_utc is None or s.limite_zeragem_utc is None:
            continue
        A = s.abertura_utc
        L = min(s.limite_zeragem_utc, s.virada_utc.replace(hour=22, minute=45))
        idx = grupos["M15"].get(d, [])
        if not idx:
            continue
        anterior = [i for i, t in enumerate(m15.fim) if int(t) <= ns(A)]
        p_anterior = int(m15.c[anterior[-1]]) if anterior else None
        for f in (30, 60, 120):
            fim_formacao = A + timedelta(minutes=f)
            form = [i for i in idx if ns(A) <= int(m15.ts[i]) and int(m15.fim[i]) <= ns(fim_formacao)]
            if p_anterior is not None and form:
                p = int(m15.c[form[-1]])
                if p != p_anterior:
                    lado = Lado.COMPRA if p > p_anterior else Lado.VENDA
                    for h in (30, 60):
                        sinal = L - timedelta(minutes=h)
                        stop = _atr(m15, sinal)
                        if stop is not None:
                            x = _intencao(f"B1|sessao|f{f}|h{h}", simbolo, lado, sinal, stop, L)
                            if x: saida.append(x)
            if not form:
                continue
            hi, lo = max(int(m15.h[i]) for i in form), min(int(m15.l[i]) for i in form)
            largura = hi - lo
            if largura <= 0:
                continue
            candidatos = [i for i in idx if int(m15.ts[i]) >= ns(fim_formacao) and int(m15.fim[i]) <= ns(L - timedelta(minutes=60))]
            cruzou = next((i for i in candidatos if int(m15.c[i]) > hi or int(m15.c[i]) < lo), None)
            if cruzou is not None:
                lado = Lado.COMPRA if int(m15.c[cruzou]) > hi else Lado.VENDA
                sinal = de_ns(int(m15.fim[cruzou]))
                for k in (0.5, 1.0):
                    x = _intencao(f"B2|sessao|f{f}|k{k:.1f}", simbolo, lado, sinal, max(1, math.ceil(k * largura)), L)
                    if x: saida.append(x)

        # B3: cada fatia atual usa só as D fatias anteriores completas.
        for minuto in range(0, max(0, int((L - A).total_seconds() // 60) - 29), 30):
            inicio = A + timedelta(minutes=minuto)
            fim = inicio + timedelta(minutes=30)
            atuais = [i for i in idx if ns(inicio) <= int(m15.ts[i]) and int(m15.fim[i]) <= ns(fim)]
            if len(atuais) != 2 or de_ns(int(m15.ts[atuais[1]])) - de_ns(int(m15.ts[atuais[0]])) != M15:
                continue
            for D in (20, 60, 120):
                completos = [
                    fatias30[(pd, minuto)]
                    for pd in dias[:di]
                    if (pd, minuto) in fatias30
                ]
                vals = completos[-D:]
                if len(vals) != D:
                    continue
                m, s_amostral = _media_desvio_exato(vals)
                if m == 0.0 or s_amostral == 0.0:
                    continue
                t = m / (s_amostral / math.sqrt(D))
                stop = _atr(m15, inicio)
                if stop is None or t == 0:
                    continue
                for q in (1, 2):
                    if abs(t) > q:
                        x = _intencao(f"B3|D{D}|q{q}", simbolo, Lado.COMPRA if t > 0 else Lado.VENDA, inicio, stop, fim)
                        if x: saida.append(x)

    # B4 e B5 por tempo gráfico.
    for tf, (b, dur) in por_tf.items():
        pg = grupos[tf]
        for di, d in enumerate(dias):
            s = sessoes[d]
            A = s.abertura_utc
            L = min(s.limite_zeragem_utc, s.virada_utc.replace(hour=22, minute=45))
            idx = pg.get(d, [])
            anteriores = dias[max(0, di - 20):di]
            retornos = []
            fontes = set(j for pd in anteriores for j in pg.get(pd, []))
            for j in sorted(fontes):
                if j > 0 and j - 1 in fontes and int(b.ts[j] - b.ts[j - 1]) == int(dur.total_seconds() * 1e9):
                    retornos.append(int(b.c[j]) / int(b.c[j - 1]) - 1.0)
            _, sigma = _media_desvio_exato(retornos) if di >= 20 else (0.0, 0.0)
            for k in (2, 3):
                if sigma > 0:
                    for H in (1, 4):
                        for j in idx:
                            if j == 0 or int(b.ts[j] - b.ts[j - 1]) != int(dur.total_seconds() * 1e9):
                                continue
                            sinal = de_ns(int(b.fim[j]))
                            if not (A <= de_ns(int(b.ts[j])) and sinal < L):
                                continue
                            ret = int(b.c[j]) / int(b.c[j - 1]) - 1.0
                            if abs(ret) > k * sigma:
                                stop = _atr(b, sinal)
                                limite = min(L, sinal + H * dur)
                                if stop is not None:
                                    x = _intencao(f"B4|{tf}|k{k}|H{H}", simbolo, Lado.VENDA if ret > 0 else Lado.COMPRA, sinal, stop, limite)
                                    if x: saida.append(x)
            closes = b.c.astype(float)
            for curta, longa in ((5, 20), (10, 50), (20, 100)):
                emitida = False
                for j in idx:
                    if emitida or j < longa:
                        continue
                    sinal = de_ns(int(b.fim[j]))
                    if not (A <= de_ns(int(b.ts[j])) and sinal <= L - timedelta(minutes=60)):
                        continue
                    agora = closes[j-curta+1:j+1].mean() - closes[j-longa+1:j+1].mean()
                    antes = closes[j-curta:j].mean() - closes[j-longa:j].mean()
                    if (antes <= 0 < agora) or (antes >= 0 > agora):
                        stop = _atr(b, sinal)
                        if stop is not None:
                            x = _intencao(f"B5|{tf}|{curta}x{longa}", simbolo, Lado.COMPRA if agora > 0 else Lado.VENDA, sinal, stop, L)
                            if x: saida.append(x); emitida = True
    return sorted(saida, key=lambda i: (i.barra_sinal_fim, i.regra_id, i.intencao_id))


__all__ = ["gerar"]
