"""F2-2: nenhuma decisão até t muda quando se altera, apaga ou acrescenta dado posterior a t."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Iterable, Sequence

import numpy as np

from titan.tipos_f2 import Intencao
from titan.visao import Barras, VisaoMemoria, ns


@dataclass(frozen=True)
class Divergencia:
    corte: datetime
    instante: datetime
    modo: str
    completa: tuple[str, ...]
    cortada: tuple[str, ...]


def _ids(intencoes: Iterable[Intencao]) -> tuple[str, ...]:
    return tuple(sorted(i.intencao_id for i in intencoes))


def _apos(b: Barras, corte: datetime) -> int:
    return int(np.searchsorted(b.fim, ns(corte), side="right"))


def _apagar(b: Barras, corte: datetime, rng: random.Random) -> Barras:
    return b.fatia(0, _apos(b, corte))


def _alterar(b: Barras, corte: datetime, rng: random.Random) -> Barras:
    k = _apos(b, corte)
    if k == len(b):
        return b
    gen = np.random.default_rng(rng.getrandbits(32))
    choque = gen.integers(-5000, 5000, size=len(b) - k)
    o, h, l, c = (x.copy() for x in (b.o, b.h, b.l, b.c))
    o[k:] += choque
    c[k:] += choque[::-1]
    h[k:] = np.maximum.reduce([h[k:] + 7000, o[k:], c[k:]])
    l[k:] = np.minimum.reduce([l[k:] - 7000, o[k:], c[k:]])
    sp = b.spread.copy()
    sp[k:] = sp[k:] * 9 + 1
    return Barras(b.tempo_grafico, b.ts, o, h, l, c, sp)


def _acrescentar(b: Barras, corte: datetime, rng: random.Random) -> Barras:
    """Apaga o depois de t e põe outra história inventada no lugar, com mais barras."""
    k = _apos(b, corte)
    passo = int(b.fim[0] - b.ts[0]) if len(b) else 0
    if k == 0 or passo == 0:
        return b
    m = (len(b) - k) + 50
    # a história inventada começa no corte: nada novo pode ter fim <= t
    ts = max(int(b.ts[k - 1]) + passo, ns(corte)) + passo * np.arange(0, m, dtype=np.int64)
    base = int(b.c[k - 1])
    c = base + np.cumsum(np.random.default_rng(rng.getrandbits(32)).integers(-300, 301, size=m))
    o = np.concatenate([[base], c[:-1]])
    h = np.maximum(o, c) + 10
    l = np.minimum(o, c) - 10
    sp = np.full(m, 3, dtype=np.int64)
    cat = lambda x, y: np.concatenate([x[:k], y]).astype(np.int64)  # noqa: E731
    return Barras(b.tempo_grafico, cat(b.ts, ts), cat(b.o, o), cat(b.h, h), cat(b.l, l), cat(b.c, c),
                  cat(b.spread, sp))


MODOS: dict[str, Callable] = {"apagar": _apagar, "alterar": _alterar, "acrescentar": _acrescentar}


def conferir_causalidade(
    decidir: Callable[[VisaoMemoria, datetime], list[Intencao]],
    visao: VisaoMemoria,
    instantes: Sequence[datetime],
    n_cortes: int = 200,
    semente: int = 8,
) -> list[Divergencia]:
    """Corta em ``n_cortes`` datas sorteadas entre os instantes; para cada corte e cada modo, as decisões
    em todo instante <= corte têm de ser idênticas às da série inteira. Devolve as divergências (vazia = ok)."""
    rng = random.Random(semente)
    instantes = sorted(instantes)
    completa = {t: _ids(decidir(visao.em(t), t)) for t in instantes}
    divergencias: list[Divergencia] = []
    for corte in sorted(rng.choice(instantes) for _ in range(n_cortes)):
        for modo, f in MODOS.items():
            dados = {k: f(b, corte, rng) for k, b in visao.dados.items()}
            v = replace(visao, dados=dados, agora=corte)
            v._fins = {}
            for t in instantes:
                if t > corte:
                    break
                ids = _ids(decidir(v.em(t), t))
                if ids != completa[t]:
                    divergencias.append(Divergencia(corte, t, modo, completa[t], ids))
    return divergencias
