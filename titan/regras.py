"""Regras da campanha 1 (day trade, M15). Núcleo = momento intradiário (Gao, Han, Li e Zhou, 2018).

Toda regra decide com ``VisaoCausal`` no instante ``agora`` (fronteira de barra de 15 min) e devolve
``Intencao``; o motor dimensiona, aplica zeragem, filtro de custo por hora, limite do dia e margem.
Convenções aqui declaradas entram no pré-registro 8 (F2-7) e não mudam depois de congeladas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np

from titan.tipos_f2 import Campanha, Intencao, Lado, VisaoCausal
from titan.visao import ns


@dataclass(frozen=True)
class PregaoAVista:
    fuso: str
    abertura: time
    fechamento: time
    cfd_fecha_junto: bool  # o CFD para no fechamento do pregão à vista (índices americanos na ActivTrades)


# Horário do pregão à vista na bolsa de origem (hora local da bolsa). Medido no calendário inferido
# (F1-13): CFD americano termina às 23:00 do servidor = 16:00 de Nova York; europeus seguem até 22:00.
# Ásia e Brasil ficam fora do Núcleo: pausa de almoço (Jp225, HKInd) ou pregão sem barra do CFD no fim.
PREGOES: dict[str, PregaoAVista] = {
    "Usa500": PregaoAVista("America/New_York", time(9, 30), time(16, 0), True),
    "UsaTec": PregaoAVista("America/New_York", time(9, 30), time(16, 0), True),
    "UsaInd": PregaoAVista("America/New_York", time(9, 30), time(16, 0), True),
    "UsaRus": PregaoAVista("America/New_York", time(9, 30), time(16, 0), True),
    "Ger40": PregaoAVista("Europe/Berlin", time(9, 0), time(17, 30), False),
    "GerMid50": PregaoAVista("Europe/Berlin", time(9, 0), time(17, 30), False),
    "GerTec": PregaoAVista("Europe/Berlin", time(9, 0), time(17, 30), False),
    "Euro50": PregaoAVista("Europe/Berlin", time(9, 0), time(17, 30), False),
    "Fra40": PregaoAVista("Europe/Paris", time(9, 0), time(17, 30), False),
    "Ita40": PregaoAVista("Europe/Rome", time(9, 0), time(17, 30), False),
    "Swi20": PregaoAVista("Europe/Zurich", time(9, 0), time(17, 30), False),
    "UK100": PregaoAVista("Europe/London", time(8, 0), time(16, 30), False),
}

M15 = timedelta(minutes=15)
AUSENTE = np.iinfo(np.int64).min


def _instante(dia: date, hora: time, fuso: str) -> datetime:
    return datetime.combine(dia, hora, ZoneInfo(fuso)).astimezone(ZoneInfo("UTC"))


@lru_cache(maxsize=200_000)
def janela_nucleo(simbolo: str, dia: date) -> tuple[datetime, datetime, datetime, datetime]:
    """(abertura+30min, entrada, saída, fechamento) em UTC no dia local da bolsa.

    Entrada na última meia hora; se o CFD para junto com o pregão, não existe barra que abra no
    fechamento para sair (emenda C7), e a janela de 30 min recua 15 min: [fech-45, fech-15].
    """
    p = PREGOES[simbolo]
    abertura = _instante(dia, p.abertura, p.fuso)
    fechamento = _instante(dia, p.fechamento, p.fuso)
    recuo = M15 if p.cfd_fecha_junto else timedelta(0)
    saida = fechamento - recuo
    entrada = saida - timedelta(minutes=30)
    return abertura + timedelta(minutes=30), entrada, saida, fechamento


@lru_cache(maxsize=200_000)
def _janela_ns(simbolo: str, dia: date) -> tuple[int, int, int, int]:
    return tuple(ns(x) for x in janela_nucleo(simbolo, dia))  # type: ignore[return-value]


@dataclass(frozen=True)
class Nucleo:
    """Lado da última meia hora = sinal do retorno do fechamento anterior até abertura+30 min."""

    simbolos: tuple[str, ...] = tuple(PREGOES)
    stop_multiplo: float = 3.0  # stop = 3 × mediana |movimento da janela| dos pregões anteriores
    stop_pregoes: int = 60
    stop_minimo_obs: int = 20
    regra_id: str = "nucleo_momento_intradiario"
    campanha: Campanha = Campanha.DAY_TRADE_M15

    def decidir(self, visao: VisaoCausal, agora: datetime) -> list[Intencao]:
        saida: list[Intencao] = []
        elegiveis = set(visao.elegiveis(self.campanha))
        for s in self.simbolos:
            if s not in elegiveis:
                continue
            dia = agora.astimezone(ZoneInfo(PREGOES[s].fuso)).date()
            abre30, entrada, fim_janela, fechamento = janela_nucleo(s, dia)
            if agora != entrada:
                continue
            intencao = self._decidir_simbolo(visao, s, dia, abre30, entrada, fim_janela)
            if intencao is not None:
                saida.append(intencao)
        return saida

    def _decidir_simbolo(self, visao, s, dia, abre30, entrada, fim_janela) -> Intencao | None:
        n = 96 * (self.stop_pregoes + 40)
        b = visao.barras(s, "M15", n)
        if len(b) == 0:
            return None
        fins = b.fim

        def fecha_em(alvos: np.ndarray) -> np.ndarray:
            """Fechamento da barra que termina em cada alvo; AUSENTE onde ela não existe (preço pode ser negativo)."""
            i = np.minimum(np.searchsorted(fins, alvos), len(fins) - 1)
            return np.where(fins[i] == alvos, b.c[i], AUSENTE)

        p_abre30 = int(fecha_em(np.array([ns(abre30)]))[0])
        if p_abre30 == AUSENTE:
            return None  # dia sem sinal: sem a barra de abertura+30 (convenção)
        k_max = 3 * self.stop_pregoes
        passado = np.array([_janela_ns(s, dia - timedelta(days=k)) for k in range(1, k_max)], dtype=np.int64)
        fech_ant = fecha_em(passado[:5, 3])  # último pregão à vista anterior, até 5 dias corridos
        achados = fech_ant[fech_ant != AUSENTE]
        if len(achados) == 0 or int(achados[0]) == p_abre30:
            return None
        p_ant = int(achados[0])
        pe, pf = fecha_em(passado[:, 1]), fecha_em(passado[:, 2])
        ok = (pe != AUSENTE) & (pf != AUSENTE)
        movimentos = np.abs(pf[ok] - pe[ok])[: self.stop_pregoes]
        if len(movimentos) < self.stop_minimo_obs:
            return None
        stop = max(1, math.ceil(self.stop_multiplo * float(np.median(movimentos))))
        return Intencao(campanha=self.campanha, regra_id=self.regra_id, simbolo=s,
                        lado=Lado.COMPRA if p_abre30 > p_ant else Lado.VENDA,
                        barra_sinal_fim=entrada, stop_pontos=stop, hora_limite=fim_janela)
