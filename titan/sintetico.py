"""Mundos sintéticos da fase 2 (F2-9 sem vantagem, F2-12 com vantagem plantada). Nunca preço real (T6, A2).

Cada mundo: N ativos de câmbio sintéticos, M15 do dia de servidor (96 barras, seg–sex), preço em pontos
inteiros, volatilidade com forma intradiária em U e regime diário log-AR(1), caudas t de Student (5 g.l.),
fator comum entre os ativos. A vantagem plantada é uma dependência causal sinal→retorno que alguma família
da gramática (parte B) consegue ver; "deriva" é a família da emenda C2 (média positiva sem previsão).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from titan.calendario import meia_noite_servidor_utc, offset_servidor_min
from titan.motor import BarraMotor, EspecificacaoAtivo
from titan.tipos import Sessao
from titan.visao import Barras, de_ns, ns

M15 = 15 * 60 * 10**9
BARRAS_DIA = 96
BARRA_L = 91  # 22:45 do servidor
TIPOS = ("nenhuma", "deriva", "momento", "reversao", "sazonal")


@dataclass(frozen=True)
class Vantagem:
    tipo: str = "nenhuma"
    forca: float = 0.0  # média plantada em desvios-padrão da barra afetada
    fracao_ativos: float = 1.0  # fração dos ativos que recebem a vantagem

    def __post_init__(self) -> None:
        if self.tipo not in TIPOS:
            raise ValueError(f"vantagem desconhecida: {self.tipo}")
        if self.tipo == "nenhuma" and self.forca != 0:
            raise ValueError("sem vantagem, força zero")


@dataclass
class Mundo:
    dias: list[date]
    m15: dict[str, Barras]
    sessoes: dict[str, dict[date, Sessao]]
    specs: dict[str, EspecificacaoAtivo]
    com_vantagem: tuple[str, ...]
    lado_plantado: dict[str, float] = field(default_factory=dict)  # sazonal: sinal plantado por ativo (T12)
    _motor: dict[str, tuple[BarraMotor, ...]] = field(default_factory=dict)

    def barras_motor(self, simbolo: str) -> tuple[BarraMotor, ...]:
        if simbolo not in self._motor:
            b = self.m15[simbolo]
            out, cache = [], {}
            for i in range(len(b)):
                t = de_ns(int(b.ts[i]))
                h = t.replace(minute=0)
                if h not in cache:  # dia de servidor pelo relógio (vale com barras apagadas ou acrescentadas)
                    cache[h] = (h + timedelta(minutes=offset_servidor_min(h))).date()
                out.append(BarraMotor(simbolo, t, "M15", int(b.o[i]), int(b.h[i]), int(b.l[i]),
                                      int(b.c[i]), int(b.spread[i]), cache[h]))
            self._motor[simbolo] = tuple(out)
        return self._motor[simbolo]

    def calendario(self, simbolo: str, dia: date) -> Sessao:
        return self.sessoes[simbolo][dia]

    def especificacao(self, simbolo: str, instante) -> EspecificacaoAtivo:
        return self.specs[simbolo]


def sharpe_ideal(mundo: "Mundo", vantagem: Vantagem) -> float:
    """Sharpe anual líquido do spread da regra que conhece a vantagem, média sobre os ativos com vantagem.

    momento: entra em L−60 no lado da formação (barras 0–1), sai em L. reversao: a regra B4 M15 k=2 H=1 —
    σ = desvio amostral dos retornos simples entre barras adjacentes dos 20 pregões anteriores (sem o retorno
    cuja origem está antes da janela); barra além de 2σ → uma barra contra, entrada até L−1 barra. sazonal:
    fatia 10:00–10:30 no lado plantado (Mundo.lado_plantado, não a amostra). deriva: comprado até L.
    """
    out = []
    for s in mundo.com_vantagem:
        b = mundo.m15[s]
        c = b.c.reshape(-1, BARRAS_DIA).astype(float)
        o = b.o.reshape(-1, BARRAS_DIA).astype(float)
        S = b.spread.reshape(-1, BARRAS_DIA).astype(float)  # spread da barra de entrada de cada operação
        if vantagem.tipo == "momento":
            lado = np.sign(c[:, 1] - o[:, 0])
            pnl = lado * (c[:, BARRA_L - 1] - c[:, BARRA_L - 5]) - S[:, BARRA_L - 4] * (lado != 0)
        elif vantagem.tipo == "sazonal":
            mov = c[:, 41] - c[:, 39]
            pnl = mundo.lado_plantado[s] * mov - S[:, 40]
        elif vantagem.tipo == "deriva":
            pnl = c[:, BARRA_L - 1] - o[:, 0] - S[:, 0]
        elif vantagem.tipo == "reversao":
            cf = b.c.astype(float)
            ret = np.full(len(cf), np.nan)
            ret[1:] = cf[1:] / cf[:-1] - 1.0
            ret[1:][np.diff(b.ts) != M15] = np.nan  # só barras adjacentes, como B4
            R = ret.reshape(-1, BARRAS_DIA)
            ok = np.isfinite(R)
            x = np.where(ok, R, 0.0)
            s1, s2, nn = x.sum(1), (x * x).sum(1), ok.sum(1).astype(float)
            # janela de 20 pregões: tira o retorno da 1ª barra do 1º dia (origem antes da janela)
            prim = np.where(ok[:, 0], x[:, 0], 0.0)
            nd = len(R)
            sig = np.full(nd, np.nan)
            for d in range(20, nd):
                a, z = d - 20, d
                S1 = s1[a:z].sum() - prim[a]
                S2 = s2[a:z].sum() - prim[a] ** 2
                N = nn[a:z].sum() - ok[a, 0]
                if N >= 2:
                    v = (S2 - S1 * S1 / N) / (N - 1)
                    sig[d] = math.sqrt(v) if v > 0 else np.nan
            j = np.arange(BARRAS_DIA - 1)[None, :]
            choque = ok[:, :-1] & (np.abs(np.where(ok, R, 0.0)[:, :-1]) > 2 * sig[:, None]) & (j < BARRA_L - 1)
            mov = c[:, 1:] - c[:, :-1]
            pnl = np.where(np.isfinite(sig), np.where(choque, -np.sign(R[:, :-1]) * mov - S[:, 1:], 0.0).sum(1), np.nan)
        else:
            return 0.0
        pnl = pnl / o[:, 0]  # em fração do preço: estável quando o preço deriva
        pnl = pnl[np.isfinite(pnl)]
        out.append(pnl.mean() / pnl.std(ddof=1) * np.sqrt(260.3) if pnl.std(ddof=1) > 0 else 0.0)
    return float(np.mean(out)) if out else 0.0


def calibrar_forca(tipo: str, sharpe_alvo: float, semente: int = 99, n_dias: int = 1240, n_ativos: int = 8,
                   **kw) -> float:
    """Força que dá `sharpe_alvo` à regra ideal (bisseção; mundo de calibração com semente própria)."""
    lo, hi = 0.0, (0.05 if tipo == "deriva" else 1.0)  # deriva atua nas 96 barras: força pequena
    s_lo = sharpe_ideal(gerar_mundo(semente, n_dias, n_ativos, Vantagem(tipo, lo), **kw), Vantagem(tipo, lo))
    s_hi = sharpe_ideal(gerar_mundo(semente, n_dias, n_ativos, Vantagem(tipo, hi), **kw), Vantagem(tipo, hi))
    if not (math.isfinite(s_lo) and math.isfinite(s_hi) and s_lo <= sharpe_alvo <= s_hi):
        raise ValueError(f"{tipo}: Sharpe {sharpe_alvo} fora de [{s_lo}, {s_hi}] (sem bracketing)")
    for _ in range(25):
        f = (lo + hi) / 2
        sh = sharpe_ideal(gerar_mundo(semente, n_dias, n_ativos, Vantagem(tipo, f), **kw), Vantagem(tipo, f))
        lo, hi = (f, hi) if sh < sharpe_alvo else (lo, f)
    return (lo + hi) / 2


def dias_uteis(inicio: date, n: int) -> list[date]:
    d, out = inicio, []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _perfil_spread() -> np.ndarray:
    """Spread relativo por barra do dia de servidor: largo na virada (hora 23 e 00, como no câmbio real, B0b),
    um pouco mais largo na madrugada asiática, 1 no resto."""
    p = np.ones(BARRAS_DIA)
    p[92:96] = 3.0  # 23:00–24:00 do servidor
    p[0:4] = 2.0  # 00:00–01:00
    p[4:28] = 1.3  # 01:00–07:00
    return p


def _spreads(semente: int, ativo: int, n_dias: int, base: int) -> np.ndarray:
    """Spread por barra (pontos inteiros >= 1): perfil intradiário × ruído log-normal (σ = 0,35) por barra.

    Gerador próprio por (semente, ativo): não desloca os sorteios de preço — os preços de um mundo não mudam.
    Com spread variável, a série pessimista (pior spread da hora em 60 pregões) difere da normal, como no real.
    """
    rng = np.random.default_rng([semente, ativo, 7])
    s = base * np.tile(_perfil_spread(), n_dias) * np.exp(rng.normal(-0.35 ** 2 / 2, 0.35, n_dias * BARRAS_DIA))
    return np.maximum(1, np.round(s)).astype(np.int64)


def _forma_u() -> np.ndarray:
    k = np.arange(BARRAS_DIA)
    return 0.6 + 0.8 * ((k - 56) / 48.0) ** 2  # mais calmo no meio da madrugada asiática do servidor


def gerar_mundo(semente: int, n_dias: int, n_ativos: int = 8, vantagem: Vantagem = Vantagem(),
                inicio: date = date(2019, 1, 7), sigma_barra: float = 6e-4, spread_pontos: int = 8,
                correlacao: float = 0.3) -> Mundo:
    rng = np.random.default_rng(semente)
    dias = dias_uteis(inicio, n_dias)
    nd = len(dias)
    forma = _forma_u()
    forma = forma / np.sqrt((forma ** 2).mean())
    # regime diário de volatilidade (log-AR(1)) comum + idiossincrático
    def regime():
        z = np.empty(nd)
        z[0] = 0.0
        e = rng.normal(0, 0.15, nd)
        for i in range(1, nd):
            z[i] = 0.95 * z[i - 1] + e[i]
        return np.exp(z - z.var() / 2)
    comum_reg = regime()
    fator = rng.standard_t(5, (nd, BARRAS_DIA)) / np.sqrt(5 / 3)
    n_com = int(round(vantagem.fracao_ativos * n_ativos)) if vantagem.tipo != "nenhuma" else 0
    sinais_sazonais = rng.choice([-1.0, 1.0], n_ativos)
    m15, sessoes, specs, com = {}, {}, {}, []
    ts_dia = np.array([ns(meia_noite_servidor_utc(d)) for d in dias], dtype=np.int64)
    ts = (ts_dia[:, None] + np.arange(BARRAS_DIA)[None, :] * M15).ravel()
    for a in range(n_ativos):
        simbolo = f"SINT{a:02d}"
        reg = comum_reg * regime()
        sig = sigma_barra * reg[:, None] * forma[None, :]  # dias × barras
        idio = rng.standard_t(5, (nd, BARRAS_DIA)) / np.sqrt(5 / 3)
        z = np.sqrt(correlacao) * fator + np.sqrt(1 - correlacao) * idio
        r = sig * z
        c_fixo = None
        if a < n_com:
            com.append(simbolo)
            f = vantagem.forca
            if vantagem.tipo == "deriva":
                r += f * sig
            elif vantagem.tipo == "momento":  # B1: formação até A+30 prevê a última hora antes de L
                s = np.sign(r[:, 0] + r[:, 1])
                r[:, BARRA_L - 4:BARRA_L] += f * sig[:, BARRA_L - 4:BARRA_L] * s[:, None]
            elif vantagem.tipo == "reversao":  # B4: barra além de 2σ é seguida de barra contrária
                # gatilho pelo σ OBSERVÁVEL da B4 (T12 do Codex): retornos simples entre barras adjacentes
                # dos 20 pregões anteriores, sem o retorno de origem fora da janela; sequencial e causal
                # sobre os FECHAMENTOS INTEIROS já materializados, como a B4 os vê (T12, 3a volta)
                base_a = 110_000.0 * (1 + 0.1 * a)
                adj0 = np.r_[False, ts_dia[1:] == ts_dia[:-1] + BARRAS_DIA * M15]  # adjacência pelo relógio
                cint = np.empty(nd * BARRAS_DIA, np.int64)
                ret = np.full((nd, BARRAS_DIA), np.nan)
                s1, s2, nn, prim = np.zeros(nd), np.zeros(nd), np.zeros(nd), np.zeros(nd)
                acc = 0.0
                for d in range(nd):
                    lim = None
                    if d >= 20:
                        a0 = d - 20
                        S1 = s1[a0:d].sum() - prim[a0]
                        S2 = s2[a0:d].sum() - prim[a0] ** 2
                        N = nn[a0:d].sum() - (1.0 if adj0[a0] else 0.0)
                        v = (S2 - S1 * S1 / N) / (N - 1) if N >= 2 else 0.0
                        lim = 2.0 * math.sqrt(v) if v > 0 else None
                    for k in range(BARRAS_DIA):
                        # sinal na barra k−1 dentro de [A, L−1 barra), com retorno observável (adjacente)
                        if lim is not None and 1 <= k < BARRA_L and math.isfinite(ret[d, k - 1]) \
                                and abs(ret[d, k - 1]) > lim:
                            r[d, k] -= f * sig[d, k] * math.copysign(1.0, ret[d, k - 1])
                        acc += r[d, k]
                        i = d * BARRAS_DIA + k
                        cint[i] = round(base_a * math.exp(acc))
                        if i > 0 and (k > 0 or adj0[d]):
                            ret[d, k] = cint[i] / cint[i - 1] - 1.0
                    x = ret[d][np.isfinite(ret[d])]
                    s1[d], s2[d], nn[d] = x.sum(), (x * x).sum(), len(x)
                    prim[d] = ret[d, 0] if adj0[d] else 0.0
                c_fixo = cint
            elif vantagem.tipo == "sazonal":  # B3: a fatia 10:00–10:30 do servidor tem sinal fixo
                r[:, 40:42] += f * sig[:, 40:42] * sinais_sazonais[a]
        base = 110_000.0 * (1 + 0.1 * a)
        c = c_fixo if c_fixo is not None else np.round(base * np.exp(np.cumsum(r.ravel()))).astype(np.int64)
        o = np.r_[int(round(base)), c[:-1]]
        amp = np.abs(rng.normal(0, 0.5, len(c))) * sig.ravel() * base
        h = np.maximum(o, c) + np.round(amp).astype(np.int64)
        l = np.minimum(o, c) - np.round(np.abs(rng.normal(0, 0.5, len(c))) * sig.ravel() * base).astype(np.int64)
        m15[simbolo] = Barras("M15", ts.copy(), o, h, l, c, _spreads(semente, a, nd, spread_pontos))
        sessoes[simbolo] = {
            d: Sessao(simbolo, d, de_ns(int(ts_dia[i])), de_ns(int(ts_dia[i]) + BARRAS_DIA * M15),
                      de_ns(int(ts_dia[i]) + BARRA_L * M15), de_ns(int(ts_dia[i]) + BARRAS_DIA * M15), 0, False,
                      "sintetico")
            for i, d in enumerate(dias)}
        specs[simbolo] = EspecificacaoAtivo(simbolo, "cambio", 1.0, 0.01, 0.01, base * 100_000 / 30 / 1e5,
                                            0.5, slippage_pontos=1)
    lado = {f"SINT{a:02d}": float(sinais_sazonais[a]) for a in range(n_ativos)} if vantagem.tipo == "sazonal" else {}
    return Mundo(dias, m15, sessoes, specs, tuple(com), {s: lado[s] for s in com} if lado else {})
