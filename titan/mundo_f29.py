"""Mundos nulos da calibração F2-9 sobre metadados reais, nunca preços reais.

O chamador entrega as sessões aprovadas por F1-13 e as especificações aprovadas
por F1-6/F1-9. O gerador copia esses contratos e calendários sem inferir nenhum
valor econômico e sintetiza somente OHLC/spread. A ausência de leitura de disco
é deliberada: a porta da História permanece fechada por T6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Mapping

import numpy as np

from titan.motor import BarraMotor, EspecificacaoAtivo
from titan.regras import PREGOES
from titan.tipos import Sessao
from titan.visao import Barras, ns

M15 = timedelta(minutes=15)
M15_NS = int(M15.total_seconds() * 1_000_000_000)
SEMENTE_INICIAL_F2_9 = 900_000


class FamiliaNula(StrEnum):
    AR_GARCH = "ar_garch"
    REGIMES_VOLATILIDADE = "regimes_volatilidade"
    REGIMES_CORRELACAO = "regimes_correlacao"
    FATORES_COMUNS = "fatores_comuns"
    GAPS = "gaps_abertura_fim_semana"
    DERIVA_POSITIVA = "deriva_positiva_sem_previsao"


@dataclass
class MundoF29:
    dias: list[date]
    m15: dict[str, Barras]
    sessoes: dict[str, dict[date, Sessao]]
    specs: dict[str, EspecificacaoAtivo]
    familia: FamiliaNula
    semente: int
    com_vantagem: tuple[str, ...] = ()
    _dias_barra: dict[str, tuple[date, ...]] = field(default_factory=dict, repr=False)
    _motor: dict[str, tuple[BarraMotor, ...]] = field(default_factory=dict, repr=False)

    def barras_motor(self, simbolo: str) -> tuple[BarraMotor, ...]:
        if simbolo not in self._motor:
            b = self.m15[simbolo]
            dias = self._dias_barra[simbolo]
            self._motor[simbolo] = tuple(
                BarraMotor(
                    simbolo,
                    datetime.fromtimestamp(int(b.ts[k]) / 1_000_000_000, tz=UTC),
                    "M15",
                    int(b.o[k]), int(b.h[k]), int(b.l[k]), int(b.c[k]),
                    int(b.spread[k]), dias[k],
                )
                for k in range(len(b))
            )
        return self._motor[simbolo]

    def calendario(self, simbolo: str, dia: date) -> Sessao:
        return self.sessoes[simbolo][dia]

    def especificacao(self, simbolo: str, instante: datetime) -> EspecificacaoAtivo:
        return self.specs[simbolo]


def _instantes(sessoes: Mapping[date, Sessao]) -> tuple[list[datetime], list[date]]:
    tempos: list[datetime] = []
    dias: list[date] = []
    for dia, sessao in sorted(sessoes.items()):
        if sessao.feriado:
            continue
        assert sessao.abertura_utc is not None and sessao.fechamento_utc is not None
        t = sessao.abertura_utc
        while t + M15 <= sessao.fechamento_utc:
            tempos.append(t)
            dias.append(dia)
            t += M15
    return tempos, dias


def _base_e_sigma(classe: str) -> tuple[float, float, int]:
    c = classe.lower()
    if "indice" in c or c in {"acao", "acoes"}:
        return 100_000.0, 7e-4, 8
    if c in {"metal", "metais"}:
        return 200_000.0, 8e-4, 12
    if c in {"energia", "energias"}:
        return 80_000.0, 1e-3, 10
    return 110_000.0, 5e-4, 3


def _markov(rng: np.random.Generator, n: int, persistencia: float = 0.97) -> np.ndarray:
    estado = np.zeros(n, dtype=np.int8)
    for k in range(1, n):
        estado[k] = estado[k - 1] if rng.random() < persistencia else 1 - estado[k - 1]
    return estado


def gerar_mundo(
    familia: FamiliaNula | str,
    semente: int,
    sessoes_reais: Mapping[str, Mapping[date, Sessao]],
    specs_reais: Mapping[str, EspecificacaoAtivo],
    *,
    inicio: date | None = None,
    fim: date | None = None,
    exigir_nucleo: bool = True,
) -> MundoF29:
    """Gera preços nulos usando literalmente calendário e contratos recebidos.

    ``fim`` é inclusivo. AR/GARCH tem dependência apenas na variância, portanto
    retorno condicional médio zero. A única média positiva é a família C2, cujo
    incremento é independente de sinais passados.
    """
    familia = FamiliaNula(familia)
    if semente < SEMENTE_INICIAL_F2_9:
        raise ValueError(f"sementes de F2-9 começam em {SEMENTE_INICIAL_F2_9}")
    comuns = set(sessoes_reais) & set(specs_reais)
    if exigir_nucleo:
        faltam = set(PREGOES) - comuns
        if faltam:
            raise ValueError(f"universo sem índices do Núcleo: {sorted(faltam)}")
    if not comuns:
        raise ValueError("universo apto vazio")

    sessoes: dict[str, dict[date, Sessao]] = {}
    for simbolo in sorted(comuns):
        escolhidas = {
            d: s for d, s in sessoes_reais[simbolo].items()
            if (inicio is None or d >= inicio) and (fim is None or d <= fim)
        }
        if escolhidas:
            sessoes[simbolo] = escolhidas
    if exigir_nucleo and set(PREGOES) - set(sessoes):
        raise ValueError("janela pedida deixa índice do Núcleo sem sessão")
    specs = {s: specs_reais[s] for s in sessoes}
    dias_globais = sorted({
        d for ss in sessoes.values() for d, sessao in ss.items()
        if not sessao.feriado
    })
    rng = np.random.default_rng(semente)

    # Fatores são indexados pelo instante UTC: ativos negociados juntos recebem
    # exatamente a mesma inovação contemporânea.
    todos_tempos = sorted({t for ss in sessoes.values() for t in _instantes(ss)[0]})
    pos_tempo = {t: k for k, t in enumerate(todos_tempos)}
    fator_1 = rng.standard_t(7, len(todos_tempos)) / np.sqrt(7 / 5)
    fator_2 = rng.normal(size=len(todos_tempos))
    estado_global = _markov(rng, len(dias_globais))
    estado_dia = dict(zip(dias_globais, estado_global, strict=True))

    m15: dict[str, Barras] = {}
    dias_barra: dict[str, tuple[date, ...]] = {}
    for numero, simbolo in enumerate(sorted(sessoes)):
        tempos, dias = _instantes(sessoes[simbolo])
        n = len(tempos)
        if not n:
            continue
        base, sigma_base, spread = _base_e_sigma(specs[simbolo].classe)
        idio = rng.standard_t(7, n) / np.sqrt(7 / 5)
        comum_1 = fator_1[[pos_tempo[t] for t in tempos]]
        comum_2 = fator_2[[pos_tempo[t] for t in tempos]]
        sigma = np.full(n, sigma_base)

        if familia is FamiliaNula.AR_GARCH:
            variancia = sigma_base ** 2
            for k in range(1, n):
                # AR(1) da variância + GARCH; nenhuma AR na média do retorno.
                variancia = 0.01 * sigma_base ** 2 + 0.08 * (
                    sigma[k - 1] * idio[k - 1]
                ) ** 2 + 0.91 * variancia
                sigma[k] = max(sigma_base * 0.25, np.sqrt(variancia))
            choque = idio
        elif familia is FamiliaNula.REGIMES_VOLATILIDADE:
            sigma *= np.array([0.55 if estado_dia[d] == 0 else 1.9 for d in dias])
            choque = idio
        elif familia is FamiliaNula.REGIMES_CORRELACAO:
            rho = np.array([0.08 if estado_dia[d] == 0 else 0.78 for d in dias])
            choque = np.sqrt(rho) * comum_1 + np.sqrt(1 - rho) * idio
        elif familia is FamiliaNula.FATORES_COMUNS:
            carga_1 = 0.35 + 0.05 * (numero % 4)
            carga_2 = 0.15 if specs[simbolo].limita_um_terco else -0.10
            resto = max(0.05, 1 - carga_1 ** 2 - carga_2 ** 2)
            choque = carga_1 * comum_1 + carga_2 * comum_2 + np.sqrt(resto) * idio
        else:
            choque = idio

        retornos = sigma * choque
        if familia is FamiliaNula.DERIVA_POSITIVA:
            retornos = retornos + 0.035 * sigma
        gaps = np.zeros(n)
        if familia is FamiliaNula.GAPS:
            for k in range(1, n):
                if tempos[k] - tempos[k - 1] > M15:
                    multiplicador = 6.0 if (dias[k] - dias[k - 1]).days >= 3 else 3.0
                    gaps[k] = multiplicador * sigma[k] * rng.standard_t(5) / np.sqrt(5 / 3)

        o = np.empty(n, dtype=np.int64)
        c = np.empty(n, dtype=np.int64)
        anterior = base
        for k in range(n):
            abertura = anterior * np.exp(gaps[k])
            fechamento = abertura * np.exp(retornos[k])
            o[k] = max(1, round(abertura))
            c[k] = max(1, round(fechamento))
            anterior = float(c[k])
        amplitude = np.maximum(
            1, np.round(base * sigma * np.abs(rng.normal(0.35, 0.18, n)))
        ).astype(np.int64)
        h = np.maximum(o, c) + amplitude
        l = np.maximum(1, np.minimum(o, c) - amplitude)
        ts = np.array([ns(t) for t in tempos], dtype=np.int64)
        m15[simbolo] = Barras(
            "M15", ts, o, h, l, c, np.full(n, spread, dtype=np.int64)
        )
        dias_barra[simbolo] = tuple(dias)

    return MundoF29(
        dias_globais, m15, sessoes, specs, familia, semente,
        _dias_barra=dias_barra,
    )


__all__ = [
    "FamiliaNula", "MundoF29", "SEMENTE_INICIAL_F2_9", "gerar_mundo",
]
