"""Gramática do Garimpo, campanha 1 (protocolo, parte B). Autor: Claude.

Gera, para um ativo, as intenções de todas as regras ao longo da série, em lote. A causalidade não é
suposta: ``conferir_causalidade_lote`` corta a série em datas sorteadas e exige que as intenções com sinal
até o corte sejam idênticas (F2-2). Retornos em fração do preço; preços em pontos inteiros.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Callable, Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from titan.calendario import servidor_para_utc
from titan.causalidade import MODOS
from titan.regras import PREGOES, Nucleo, janela_nucleo
from titan.tipos import Sessao
from titan.tipos_f2 import Campanha, Intencao, Lado
from titan.visao import Barras, ns

MIN = 60 * 1_000_000_000
M15_NS, H1_NS = 15 * MIN, 60 * MIN
AUSENTE = np.iinfo(np.int64).min
HORA_MAX_SERVIDOR = time(22, 45)  # B0b: fora da hora 23 do servidor


def agregar_h1(m15: Barras) -> Barras:
    """H1 a partir de M15: só horas com as quatro barras de 15 min (hora completa)."""
    if len(m15) == 0:
        return Barras("H1", *(np.empty(0, np.int64) for _ in range(6)))
    hora = m15.ts - (m15.ts % H1_NS)
    inicio = np.flatnonzero(np.r_[True, hora[1:] != hora[:-1]])
    fim = np.r_[inicio[1:], len(hora)]
    completas = (fim - inicio) == 4
    i0, i1 = inicio[completas], fim[completas]
    return Barras("H1", hora[i0], m15.o[i0], np.maximum.reduceat(m15.h, inicio)[completas],
                  np.minimum.reduceat(m15.l, inicio)[completas], m15.c[i1 - 1], m15.spread[i1 - 1])


@dataclass(frozen=True)
class Ref:
    """Referência de um dia: abertura A, limite L e fechamento da sessão anterior."""

    nome: str
    dia: date
    a: int
    l: int
    fech_ant: int | None


def referencias(simbolo: str, sessoes: Mapping[date, Sessao]) -> list[Ref]:
    out: list[Ref] = []
    anterior: Sessao | None = None
    for d in sorted(sessoes):
        s = sessoes[d]
        if s.feriado or s.abertura_utc is None:
            continue
        lim = min(ns(s.limite_zeragem_utc), ns(servidor_para_utc(datetime.combine(d, HORA_MAX_SERVIDOR))))
        if lim > ns(s.abertura_utc):
            out.append(Ref("sessao", d, ns(s.abertura_utc), lim,
                           ns(anterior.fechamento_utc) if anterior is not None else None))
        anterior = s
    if simbolo in PREGOES:  # referência "à vista" (A11b), mesmo recuo do Núcleo
        dias = sorted(sessoes)
        for k, d in enumerate(dias):
            a30, _, saida, _ = janela_nucleo(simbolo, d)
            ant = janela_nucleo(simbolo, dias[k - 1])[3] if k else None
            out.append(Ref("avista", d, ns(a30) - 30 * MIN, ns(saida), ns(ant) if ant else None))
    return out


class _Serie:
    """Acesso por instante às barras de um tempo gráfico."""

    def __init__(self, b: Barras):
        self.b = b
        self.fim = b.fim
        tr = np.maximum(b.h[1:], b.c[:-1]) - np.minimum(b.l[1:], b.c[:-1])
        self.tr = np.r_[b.h[:1] - b.l[:1], tr].astype(float)
        self.tr_acum = np.r_[0.0, np.cumsum(self.tr)]

    def idx_fim(self, t: int) -> int:
        """Índice da barra que termina exatamente em t; -1 se não existe."""
        i = int(np.searchsorted(self.fim, t))
        return i if i < len(self.fim) and self.fim[i] == t else -1

    def fecha(self, t: int) -> int:
        i = self.idx_fim(t)
        return AUSENTE if i < 0 else int(self.b.c[i])

    def ate(self, t: int) -> int:
        """Quantas barras têm fim <= t."""
        return int(np.searchsorted(self.fim, t, side="right"))

    def atr20(self, t: int) -> float | None:
        """B0c: média simples do true range das 20 barras com fim <= t."""
        k = self.ate(t)
        if k < 20:
            return None
        return float(self.tr_acum[k] - self.tr_acum[k - 20]) / 20.0


def _dt(t: int) -> datetime:
    from titan.visao import de_ns
    return de_ns(t)


class Bruta(NamedTuple):
    """Intenção crua da gramática, sem hash nem datetime (barata); vira Intencao ou linha de lote compacto."""

    regra: str
    simbolo: str
    lado: Lado
    sinal: int  # ns UTC, fim da barra do sinal
    stop: int  # pontos, já arredondado para cima e >= 1
    limite: int  # ns UTC, hora limite (fronteira de 15 min)

    def intencao(self) -> Intencao:
        return Intencao(Campanha.DAY_TRADE_M15, self.regra, self.simbolo, self.lado, _dt(self.sinal), self.stop,
                        _dt(self.limite))


def _intencao(regra: str, simbolo: str, lado: Lado, sinal: int, stop: float, limite: int,
              tf_ns: int = M15_NS) -> Bruta | None:
    if limite - sinal < M15_NS or stop is None or not math.isfinite(stop) or stop <= 0:
        return None  # a entrada é na barra seguinte; sem tempo até L, não há intenção
    return Bruta(regra, simbolo, lado, int(sinal), max(1, math.ceil(stop)), int(limite))


def _stop_atr(s: _Serie, t: int) -> float | None:
    a = s.atr20(t)
    return None if a is None else 3.0 * a


# ---------------------------------------------------------------- famílias

def b1_momento(simbolo: str, s: _Serie, refs: Sequence[Ref]) -> list[Intencao]:
    out = []
    for r in refs:
        if r.fech_ant is None:
            continue
        p0 = s.fecha(r.fech_ant)
        for f in (30, 60, 120):
            p1 = s.fecha(r.a + f * MIN)
            if p0 == AUSENTE or p1 == AUSENTE or p1 == p0:
                continue
            for h in (30, 60):
                t = r.l - h * MIN
                if t < r.a + f * MIN:
                    continue
                i = _intencao(f"B1|{r.nome}|f{f}|h{h}", simbolo, Lado.COMPRA if p1 > p0 else Lado.VENDA,
                              t, _stop_atr(s, t), r.l)
                if i:
                    out.append(i)
    return out


def b2_faixa(simbolo: str, s: _Serie, refs: Sequence[Ref]) -> list[Intencao]:
    out = []
    b = s.b
    for r in refs:
        i0 = int(np.searchsorted(b.ts, r.a))
        for f in (30, 60, 120):
            fim_faixa = r.a + f * MIN
            i1 = int(np.searchsorted(b.ts, fim_faixa))
            if i1 - i0 != f // 15:
                continue  # faixa incompleta
            hi, lo = int(b.h[i0:i1].max()), int(b.l[i0:i1].min())
            if hi <= lo:
                continue
            j1 = s.ate(r.l - 60 * MIN)
            for j in range(i1, j1):
                if b.c[j] > hi or b.c[j] < lo:
                    lado = Lado.COMPRA if b.c[j] > hi else Lado.VENDA
                    for k in (0.5, 1.0):
                        i = _intencao(f"B2|{r.nome}|f{f}|k{k}", simbolo, lado, int(s.fim[j]), k * (hi - lo), r.l)
                        if i:
                            out.append(i)
                    break
    return out


def _media_desvio(x) -> tuple[float, float]:
    """Média e desvio amostral por soma exatamente arredondada (math.fsum): o resultado não depende da ordem
    de soma nem da biblioteca, e o sinal da média perto de zero é único (T12 do Codex na F2-7)."""
    xs = [float(v) for v in x]
    n = len(xs)
    m = math.fsum(xs) / n
    return m, math.sqrt(math.fsum((v - m) ** 2 for v in xs) / (n - 1)) if n > 1 else 0.0


def _fecha_vet(s: _Serie, t: np.ndarray) -> np.ndarray:
    i = np.searchsorted(s.fim, t)
    ic = np.minimum(i, len(s.fim) - 1)
    ok = (i < len(s.fim)) & (s.fim[ic] == t) if len(s.fim) else np.zeros(t.shape, bool)
    return np.where(ok, s.b.c[ic] if len(s.fim) else 0, AUSENTE)


def b3_sazonal(simbolo: str, s: _Serie, refs: Sequence[Ref]) -> list[Intencao]:
    """B3 vetorizada; mesma saída de b3_sazonal_ref (teste de regressão em testes/test_gramatica.py)."""
    from numpy.lib.stride_tricks import sliding_window_view
    out = []
    sess = [r for r in refs if r.nome == "sessao"]
    if not sess:
        return out
    fat = 30 * MIN
    nslots = np.array([(r.l - fat - r.a) // fat + 1 if r.l - fat - r.a >= 0 else 0 for r in sess])
    S = int(nslots.max())
    if S <= 0:
        return out
    a = np.array([r.a for r in sess], dtype=np.int64)
    desl = np.arange(S, dtype=np.int64) * fat
    t0 = a[:, None] + desl[None, :]
    ativo = np.arange(S)[None, :] < nslots[:, None]
    p0, p1 = _fecha_vet(s, t0), _fecha_vet(s, t0 + fat)
    ok = ativo & (p0 != AUSENTE) & (p1 != AUSENTE) & (p0 != 0)
    ret = np.full(t0.shape, np.nan)
    ret[ok] = p1[ok].astype(float) / p0[ok].astype(float) - 1.0
    for j in range(S):
        v = np.flatnonzero(ok[:, j])
        x = ret[v, j]
        dias_j = np.flatnonzero(ativo[:, j])
        cnt = np.searchsorted(v, dias_j)  # retornos válidos de dias anteriores (o do próprio dia entra depois)
        for D in (20, 60, 120):
            if len(x) < D:
                continue
            sel = cnt >= D
            ks, wi = dias_j[sel], cnt[sel] - D
            ms = [_media_desvio(x[a:a + D]) for a in wi.tolist()]  # soma exata: independe da ordem (T12/F2-7)
            m = np.array([u for u, _ in ms]) if ms else np.empty(0)
            sd = np.array([v for _, v in ms]) if ms else np.empty(0)
            boa = sd != 0
            with np.errstate(divide="ignore", invalid="ignore"):
                tst = m / (sd / math.sqrt(D))
            for q in (1, 2):  # q=0 retirado (t51, acordo Claude+Codex): "sempre opera", 75% das intenções
                for k in ks[boa & (np.abs(tst) > q) & (m != 0)]:
                    pos = int(np.searchsorted(ks, k))
                    t = int(t0[k, j])
                    i = _intencao(f"B3|D{D}|q{q}", simbolo, Lado.COMPRA if m[pos] > 0 else Lado.VENDA,
                                  t, _stop_atr(s, t), t + fat)
                    if i:
                        out.append(i)
    return out


def b3_sazonal_ref(simbolo: str, s: _Serie, refs: Sequence[Ref]) -> list[Intencao]:
    """Versão de referência, literal ao texto B3 (lenta); só para o teste de regressão."""
    out = []
    sess = [r for r in refs if r.nome == "sessao"]
    hist: dict[int, list[float]] = {}  # deslocamento da fatia em relação a A -> retornos dos dias anteriores
    for r in sess:
        for desl in range(0, r.l - 30 * MIN - r.a + 1, 30 * MIN):
            t = r.a + desl
            passado = hist.get(desl, [])
            for D in (20, 60, 120):
                if len(passado) < D:
                    continue
                m, sd = _media_desvio(passado[-D:])
                if sd == 0:
                    continue
                tst = m / (sd / math.sqrt(D))
                for q in (1, 2):
                    if abs(tst) > q and m != 0:
                        i = _intencao(f"B3|D{D}|q{q}", simbolo, Lado.COMPRA if m > 0 else Lado.VENDA,
                                      t, _stop_atr(s, t), t + 30 * MIN)
                        if i:
                            out.append(i)
        # só depois de decidir o dia, entram os retornos do dia (causal)
        for desl in range(0, r.l - 30 * MIN - r.a + 1, 30 * MIN):
            p0, p1 = s.fecha(r.a + desl), s.fecha(r.a + desl + 30 * MIN)
            if p0 != AUSENTE and p1 != AUSENTE and p0 != 0:
                hist.setdefault(desl, []).append(p1 / p0 - 1.0)
    return out


def b4_reversao(simbolo: str, series: Mapping[str, _Serie], refs: Sequence[Ref]) -> list[Intencao]:
    out = []
    sess = [r for r in refs if r.nome == "sessao"]
    for tf, passo in (("M15", M15_NS), ("H1", H1_NS)):
        s = series[tf]
        b = s.b
        ret = np.full(len(b), np.nan)
        ret[1:] = b.c[1:] / np.where(b.c[:-1] == 0, np.nan, b.c[:-1]) - 1.0
        ret[1:][np.diff(b.ts) != passo] = np.nan  # só barras adjacentes (T12 do Codex, 16/11)
        for k_ref, r in enumerate(sess):
            if k_ref < 20:
                continue
            ini = sess[k_ref - 20].a
            j0, j1 = int(np.searchsorted(b.ts, ini)), int(np.searchsorted(b.ts, r.a))
            x = ret[j0 + 1:j1]  # o retorno da barra j0 começa antes da janela: fora
            x = x[np.isfinite(x)]
            if len(x) < 2:
                continue
            sig = _media_desvio(x)[1]
            if sig == 0:
                continue
            for j in range(int(np.searchsorted(b.ts, r.a)), s.ate(r.l)):
                t = int(s.fim[j])
                if not np.isfinite(ret[j]):
                    continue
                for k in (2, 3):
                    if abs(ret[j]) > k * sig:
                        lado = Lado.VENDA if ret[j] > 0 else Lado.COMPRA
                        for H in (1, 4):
                            i = _intencao(f"B4|{tf}|k{k}|H{H}", simbolo, lado, t, _stop_atr(s, t),
                                          min(t + H * passo, r.l))
                            if i:
                                out.append(i)
    return out


def b5_medias(simbolo: str, series: Mapping[str, _Serie], refs: Sequence[Ref]) -> list[Intencao]:
    out = []
    sess = [r for r in refs if r.nome == "sessao"]
    for tf in ("M15", "H1"):
        s = series[tf]
        c = s.b.c.astype(float)
        acum = np.r_[0.0, np.cumsum(c)]
        for curta, longa in ((5, 20), (10, 50), (20, 100)):
            n = len(c)
            if n <= longa:
                continue
            idx = np.arange(longa, n + 1)  # média das barras [i-janela, i)
            mc = (acum[idx] - acum[idx - curta]) / curta
            ml = (acum[idx] - acum[idx - longa]) / longa
            estado = np.sign(mc - ml)  # estado ao fim da barra idx-1
            fim_barra = s.fim[idx - 1]
            cruz = np.flatnonzero((estado[1:] != estado[:-1]) & (estado[1:] != 0)) + 1
            tempos, lados = fim_barra[cruz], estado[cruz]
            for r in sess:
                a, b = np.searchsorted(tempos, r.a, side="right"), np.searchsorted(tempos, r.l - 60 * MIN, side="right")
                if a < b:  # primeiro cruzamento do dia (B5)
                    t = int(tempos[a])
                    i = _intencao(f"B5|{tf}|{curta}x{longa}", simbolo,
                                  Lado.COMPRA if lados[a] > 0 else Lado.VENDA, t, _stop_atr(s, t), r.l)
                    if i:
                        out.append(i)
    return out


class _VisaoLote:
    """Visão causal mínima de um ativo para o Núcleo em lote: barras com fim <= instante."""

    def __init__(self, m15: Barras):
        self.m15, self.t = m15, 0

    def em(self, instante: datetime) -> "_VisaoLote":
        self.t = ns(instante)
        return self

    def barras(self, simbolo: str, tempo_grafico: str, n: int) -> Barras:
        k = int(np.searchsorted(self.m15.fim, self.t, side="right"))
        return self.m15.fatia(max(0, k - n), k)


def b6_nucleo(simbolo: str, m15: Barras, sessoes: Mapping[date, Sessao]) -> list[Bruta]:
    """B6: a regra do Núcleo (A11b), literal — a mesma classe, o mesmo regra_id —, nos índices de PREGOES.

    Assim o Garimpo contém o Núcleo de fato (D3, T12 do Codex). Um dia por data de sessão.
    """
    if simbolo not in PREGOES:
        return []
    nuc, vis, out = Nucleo(simbolos=(simbolo,)), _VisaoLote(m15), []
    for d in sorted(sessoes):
        abre30, entrada, fim_janela, _ = janela_nucleo(simbolo, d)
        i = nuc._decidir_simbolo(vis.em(entrada), simbolo, d, abre30, entrada, fim_janela)
        if i is not None:
            out.append(Bruta(i.regra_id, i.simbolo, i.lado, ns(i.barra_sinal_fim), i.stop_pontos, ns(i.hora_limite)))
    return out


def gerar_brutas(simbolo: str, m15: Barras, sessoes: Mapping[date, Sessao]) -> list[Bruta]:
    """Todas as intenções cruas de todas as regras, sem deduplicar nem ordenar."""
    refs = referencias(simbolo, sessoes)
    series = {"M15": _Serie(m15), "H1": _Serie(agregar_h1(m15))}
    return (b1_momento(simbolo, series["M15"], refs) + b2_faixa(simbolo, series["M15"], refs)
            + b3_sazonal(simbolo, series["M15"], refs) + b4_reversao(simbolo, series, refs)
            + b5_medias(simbolo, series, refs) + b6_nucleo(simbolo, m15, sessoes))


def gerar_compacto(simbolo: str, m15: Barras, sessoes: Mapping[date, Sessao]):
    """As mesmas intenções de gerar(), como LoteIntencoesCompactas (contrato do Codex), sem criar Intencao."""
    from titan.intencoes_compactas import LoteIntencoesCompactas
    brutas = gerar_brutas(simbolo, m15, sessoes)
    tabela = regras_do_ativo(simbolo)
    pos = {r: i for i, r in enumerate(tabela)}
    n = len(brutas)
    return LoteIntencoesCompactas.criar(
        Campanha.DAY_TRADE_M15, simbolo, tabela,
        np.fromiter((pos[b.regra] for b in brutas), np.uint16, n),
        np.fromiter((1 if b.lado is Lado.COMPRA else -1 for b in brutas), np.int8, n),
        np.fromiter((b.sinal for b in brutas), np.int64, n),
        np.fromiter((b.stop for b in brutas), np.int64, n),
        np.fromiter((b.limite for b in brutas), np.int64, n),
        np.zeros(n, np.int64), np.ones(n, np.float64), canonicalizar=True)


def gerar(simbolo: str, m15: Barras, sessoes: Mapping[date, Sessao]) -> list[Intencao]:
    """Todas as intenções de todas as regras da gramática para um ativo."""
    out = [b.intencao() for b in gerar_brutas(simbolo, m15, sessoes)]
    vistos: dict[str, Intencao] = {}
    for i in out:
        vistos.setdefault(i.intencao_id, i)
    return sorted(vistos.values(), key=lambda i: (i.barra_sinal_fim, i.intencao_id))


def regras_do_ativo(simbolo: str) -> tuple[str, ...]:
    refs = ("sessao", "avista") if simbolo in PREGOES else ("sessao",)
    ids = [f"B1|{r}|f{f}|h{h}" for r in refs for f in (30, 60, 120) for h in (30, 60)]
    ids += [f"B2|{r}|f{f}|k{k}" for r in refs for f in (30, 60, 120) for k in (0.5, 1.0)]
    ids += [f"B3|D{D}|q{q}" for D in (20, 60, 120) for q in (1, 2)]
    ids += [f"B4|{tf}|k{k}|H{H}" for tf in ("M15", "H1") for k in (2, 3) for H in (1, 4)]
    ids += [f"B5|{tf}|{a}x{b}" for tf in ("M15", "H1") for a, b in ((5, 20), (10, 50), (20, 100))]
    if simbolo in PREGOES:
        ids.append(Nucleo.regra_id)  # B6
    return tuple(ids)


def conferir_causalidade_lote(simbolo: str, m15: Barras, sessoes: Mapping[date, Sessao],
                              n_cortes: int = 200, semente: int = 8,
                              gerador: Callable = gerar) -> list[tuple[datetime, str, int, int]]:
    """F2-2 em lote: intenções com sinal até o corte são idênticas com o depois apagado, alterado ou trocado."""
    rng = random.Random(semente)
    cheia = gerador(simbolo, m15, sessoes)
    tempos = sorted({i.barra_sinal_fim for i in cheia})
    if not tempos:
        return []
    falhas = []
    for corte in sorted(rng.choice(tempos) for _ in range(n_cortes)):
        esperado = {i.intencao_id for i in cheia if i.barra_sinal_fim <= corte}
        for modo, f in MODOS.items():
            outra = gerador(simbolo, f(m15, corte, rng), sessoes)
            obtido = {i.intencao_id for i in outra if i.barra_sinal_fim <= corte}
            if obtido != esperado:
                falhas.append((corte, modo, len(esperado ^ obtido), len(esperado)))
    return falhas
