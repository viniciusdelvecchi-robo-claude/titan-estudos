"""Visão causal em memória (F2-2): uma regra no instante ``agora`` só enxerga barras com fim <= agora.

Os arrays vêm do Carregador (modo sintético ou cego antes do ato). Nada aqui lê disco.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Callable, Mapping

import numpy as np

from titan.tipos import Sessao, _utc_consciente
from titan.tipos_f2 import Campanha

DURACAO = {"M15": timedelta(minutes=15), "H1": timedelta(hours=1), "D1": timedelta(days=1)}
_NS = 1_000_000_000


def ns(instante: datetime) -> int:
    _utc_consciente(instante, "instante")
    return int(instante.timestamp()) * _NS + instante.microsecond * 1000


def de_ns(valor: int) -> datetime:
    return datetime.fromtimestamp(valor // _NS, tz=UTC).replace(microsecond=(valor % _NS) // 1000)


@dataclass(frozen=True)
class Barras:
    """Colunas alinhadas; ``ts`` = abertura da barra em ns UTC; preços e spread em pontos inteiros."""

    tempo_grafico: str
    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    spread: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.ts)
        if any(len(x) != n for x in (self.o, self.h, self.l, self.c, self.spread)):
            raise ValueError("colunas de tamanhos diferentes")
        if n > 1 and not np.all(np.diff(self.ts) > 0):
            raise ValueError("barras fora de ordem ou repetidas")

    @property
    def fim(self) -> np.ndarray:
        return self.ts + int(DURACAO[self.tempo_grafico].total_seconds()) * _NS

    def fatia(self, a: int, b: int) -> "Barras":
        return Barras(self.tempo_grafico, self.ts[a:b], self.o[a:b], self.h[a:b], self.l[a:b], self.c[a:b],
                      self.spread[a:b])

    def __len__(self) -> int:
        return len(self.ts)

    @classmethod
    def de_tabela(cls, tempo_grafico: str, tabela) -> "Barras":
        ts = tabela["ts_utc"].cast("int64").to_numpy() * 1000  # us -> ns
        col = lambda n: tabela[n].to_numpy().astype(np.int64)  # noqa: E731
        return cls(tempo_grafico, ts, col("open_points"), col("high_points"), col("low_points"),
                   col("close_points"), col("spread_points"))


class FuturoRecusado(PermissionError):
    pass


@dataclass
class VisaoMemoria:
    """Implementa ``tipos_f2.VisaoCausal`` sobre arrays em memória."""

    dados: Mapping[tuple[str, str], Barras]
    agora: datetime
    sessoes: Mapping[str, Mapping[date, Sessao]] = field(default_factory=dict)
    aptidao: Callable[[str, str, datetime], bool] = lambda s, u, t: True
    contratos: Callable[[str, date], str] = lambda base, d: base
    universo: Mapping[Campanha, tuple[str, ...]] = field(default_factory=dict)
    _fins: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _utc_consciente(self.agora, "agora")

    def em(self, agora: datetime) -> "VisaoMemoria":
        v = VisaoMemoria(self.dados, agora, self.sessoes, self.aptidao, self.contratos, self.universo)
        v._fins = self._fins
        return v

    def _fim(self, chave) -> np.ndarray:
        if chave not in self._fins:
            self._fins[chave] = self.dados[chave].fim
        return self._fins[chave]

    def barras(self, simbolo: str, tempo_grafico: str, n: int) -> Barras:
        chave = (simbolo, tempo_grafico)
        if chave not in self.dados:
            return Barras(tempo_grafico, *(np.empty(0, np.int64) for _ in range(6)))
        b = int(np.searchsorted(self._fim(chave), ns(self.agora), side="right"))
        return self.dados[chave].fatia(max(0, b - n), b)

    def apto(self, simbolo: str, uso: str) -> bool:
        return bool(self.aptidao(simbolo, uso, self.agora))

    def sessao(self, simbolo: str, data_servidor: date) -> Sessao:
        # O calendário é horário publicado; conhecer a sessão de hoje não revela preço.
        return self.sessoes[simbolo][data_servidor]

    def contrato(self, simbolo_base: str, data_servidor: date) -> str:
        if datetime(data_servidor.year, data_servidor.month, data_servidor.day, tzinfo=UTC) > self.agora + timedelta(days=1):
            raise FuturoRecusado("contrato de data futura")
        return self.contratos(simbolo_base, data_servidor)

    def elegiveis(self, campanha: Campanha) -> tuple[str, ...]:
        return tuple(s for s in self.universo.get(campanha, ()) if self.apto(s, campanha.value))
