"""Contratos comuns da fase 2 (pré-registro 8), acordados por Claude e Codex em 16/11/2026 (mesa t10).

Fluxo: Regra.decidir(VisaoCausal, t) -> Intencao; Motor.simular(...) -> ResultadoMotor;
Prova.testar([EntradaProva], ParamsPortao) -> VereditoPortao. O Garimpo chama o motor, nunca o contrário.
Instantes em UTC com fuso explícito; preços, stops e spreads em pontos inteiros.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Mapping, Protocol, Sequence

from titan.tipos import Sessao, _utc_consciente

CAPITAL_PETREO_USD = 3000.0
DERRAPAGEM_MAX_PONTOS = 50  # convenção comum: acima disso a execução não confere


class Campanha(StrEnum):
    DAY_TRADE_M15 = "c1_day_trade_m15"
    DIARIA = "c2_diaria"


class Lado(StrEnum):
    COMPRA = "compra"
    VENDA = "venda"


class MotivoRecusa(StrEnum):
    """Enum fechado. Cada recusa traz a desigualdade aplicada em ``dados``."""

    ZERAGEM = "zeragem"  # F2-13: não cabe antes da hora limite/virada
    FILTRO_HORA = "filtro_custo_hora"  # F2-14
    LIMITE_DIA = "limite_perda_dia"  # F2-15 / emenda C6 (igualdade recusa)
    MARGEM = "margem"  # F2-4b / emenda C8
    UM_TERCO = "um_terco_indices_acoes"  # F2-4c
    RISCO_ABERTO = "risco_aberto_alvo"  # F2-4
    PISO_F = "liquidacao_abaixo_do_piso"  # F2-4: L < F
    REDUCAO = "reducao_obrigatoria"  # F2-4
    REVOGADO = "revogado"  # F2-4
    EXPOSICAO_DESCONHECIDA = "exposicao_desconhecida"  # seção 4
    LOTE_MINIMO = "lote_minimo"
    INAPTO = "ativo_inapto"  # F2-3c / Validador
    SEM_SESSAO = "sem_sessao"
    POSICAO_ABERTA = "posicao_aberta_no_simbolo"  # conta NETTING: uma posicao por simbolo


class EstadoRisco(StrEnum):
    NORMAL = "normal"
    REDUCAO = "reducao_obrigatoria"
    DESCONHECIDA = "exposicao_desconhecida"
    REVOGADO = "revogado"


class VisaoCausal(Protocol):
    """Tudo que uma regra ou o Garimpo pode ver no instante ``agora``: só o que existia até ele.

    Cobre barras, aptidão, calendário, contratos e elegibilidade (ajuste 1 do Codex). Pedido além de
    ``agora`` levanta erro; o teste F2-2 ainda corta a série em 200 datas e compara as decisões.
    """

    agora: datetime

    def barras(self, simbolo: str, tempo_grafico: str, n: int) -> Any: ...  # últimas n com fim <= agora
    def apto(self, simbolo: str, uso: str) -> bool: ...
    def sessao(self, simbolo: str, data_servidor: date) -> Sessao: ...
    def contrato(self, simbolo_base: str, data_servidor: date) -> str: ...
    def elegiveis(self, campanha: Campanha) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class Intencao:
    campanha: Campanha
    regra_id: str
    simbolo: str
    lado: Lado
    barra_sinal_fim: datetime  # fim da barra que gerou o sinal; entrada na abertura da seguinte (F2-3)
    stop_pontos: int
    hora_limite: datetime  # zeragem: alinhada a fronteira de 15 min (emenda C7)
    alvo_pontos: int | None = None
    intencao_id: str = ""  # determinístico; preenchido se vazio (ajuste 2 do Codex)
    peso_admissao: float = 1.0  # C1: maior peso entra primeiro; não muda a identidade

    def __post_init__(self) -> None:
        _utc_consciente(self.barra_sinal_fim, "barra_sinal_fim")
        _utc_consciente(self.hora_limite, "hora_limite")
        if self.stop_pontos <= 0:
            raise ValueError("toda intenção tem stop positivo")
        if self.alvo_pontos is not None and self.alvo_pontos <= 0:
            raise ValueError("alvo, quando houver, é positivo")
        if self.hora_limite <= self.barra_sinal_fim:
            raise ValueError("hora limite tem de ser posterior ao sinal")
        if self.hora_limite.minute % 15 or self.hora_limite.second or self.hora_limite.microsecond:
            raise ValueError("hora limite alinhada a fronteira de 15 minutos (emenda C7)")
        if not self.regra_id or not self.simbolo:
            raise ValueError("regra_id e simbolo são obrigatórios")
        if math.isnan(self.peso_admissao) or self.peso_admissao < 0:  # +inf vale (V1 com variância 0)
            raise ValueError("peso de admissão não pode ser NaN nem negativo")
        if not self.intencao_id:
            chave = "|".join((self.campanha, self.regra_id, self.simbolo, self.lado,
                              self.barra_sinal_fim.isoformat(), str(self.stop_pontos), str(self.alvo_pontos),
                              self.hora_limite.isoformat()))
            object.__setattr__(self, "intencao_id", hashlib.sha256(chave.encode()).hexdigest()[:20])


class Regra(Protocol):
    regra_id: str
    campanha: Campanha

    def decidir(self, visao: VisaoCausal, agora: datetime) -> list[Intencao]: ...


@dataclass(frozen=True, slots=True)
class ParamsMotor:
    """Imutável; nenhum padrão para parâmetro do portão (F2-1)."""

    capital_usd: float
    risco_por_operacao: float  # 0,02
    risco_aberto_alvo: float  # fração do orçamento M×H (seção 4)
    limiar_m: float  # 0,20 / 0,30 / 0,50
    fracao_um_terco: float  # 1/3 (F2-4c)
    margem_max: float  # 0,50 (F2-4b)
    limite_perda_dia: float  # 0,03 (F2-15)
    filtro_spread_mov: float  # 0,20 (F2-14)
    filtro_janela_pregoes: int  # 60 (F2-14)
    pessimista: bool  # custo do lado ruim da lista T3
    capital_constante: bool = True  # True: serie de prova; False: sistema completo (F2-10)

    def __post_init__(self) -> None:
        if self.capital_usd != CAPITAL_PETREO_USD:
            raise ValueError("o capital da prova é o capital pétreo, US$ 3.000 (F2-1, F2-1b)")
        if self.limiar_m not in (0.20, 0.30, 0.50):
            raise ValueError("limiar M é 20%, 30% ou 50%")
        for nome in ("risco_por_operacao", "risco_aberto_alvo", "fracao_um_terco", "margem_max",
                     "limite_perda_dia", "filtro_spread_mov"):
            v = getattr(self, nome)
            if not 0 < v <= 1:
                raise ValueError(f"{nome} fora de (0, 1]")
        if self.filtro_janela_pregoes <= 0:
            raise ValueError("janela do filtro positiva")


@dataclass(frozen=True, slots=True)
class Recusa:
    intencao_id: str
    instante: datetime
    motivo: MotivoRecusa
    dados: Mapping[str, Any]  # lados da desigualdade aplicada


@dataclass(frozen=True, slots=True)
class ResultadoMotor:
    """Saída do motor (ajuste 3 do Codex)."""

    datas: tuple[date, ...]
    pnl_usd: tuple[float, ...]  # por dia de servidor
    retorno: tuple[float, ...]  # pnl_usd / 3.000
    exposicao_comprada: tuple[float, ...]  # para o diagnóstico de deriva (emenda C2)
    exposicao_vendida: tuple[float, ...]
    blocos: tuple[str, ...]  # rótulo do bloco independente de cada aposta (F2-8)
    livro: tuple[Mapping[str, Any], ...]
    recusas: tuple[Recusa, ...]
    estados: tuple[tuple[datetime, EstadoRisco], ...]
    conferencias_margem: tuple[Mapping[str, Any], ...]  # antes/depois de ordem e por barra (C8)

    def __post_init__(self) -> None:
        n = len(self.datas)
        if not (len(self.pnl_usd) == len(self.retorno) == len(self.exposicao_comprada)
                == len(self.exposicao_vendida) == n):
            raise ValueError("séries diárias com comprimentos diferentes")
        for p, r in zip(self.pnl_usd, self.retorno):
            if abs(p / CAPITAL_PETREO_USD - r) > 1e-12:
                raise ValueError("retorno tem de ser pnl / US$ 3.000 (F2-1b)")


@dataclass(frozen=True, slots=True)
class ParamsPortao:
    """F2-1: sem padrão; os críticos saem daqui."""

    fatia: float  # 0,025 na campanha 1
    folga: float  # 0,80
    passos_holm: int  # 2
    apostas_min: int  # 30 blocos (F2-8)
    nivel_cofre: float  # 0,05 unilateral
    revogacao_max: float  # 0,20 das partidas em 3 anos
    reamostragens: int
    semente: int

    def niveis(self) -> tuple[float, ...]:
        """Níveis operacionais dos passos de Holm: fatia/k, fatia/(k-1), ..., com a folga."""
        return tuple(self.folga * self.fatia / (self.passos_holm - i) for i in range(self.passos_holm))

    def __post_init__(self) -> None:
        if not 0 < self.fatia <= 0.05 or not 0 < self.folga <= 1 or self.passos_holm < 1:
            raise ValueError("fatia, folga ou passos inválidos")
        if self.apostas_min < 1 or self.reamostragens < 1:
            raise ValueError("mínimos positivos")


@dataclass(frozen=True, slots=True)
class EntradaProva:
    """Um candidato (Núcleo ou Garimpo) como entra na prova (ajuste 4 do Codex)."""

    candidato: str
    normal: ResultadoMotor
    pessimista: ResultadoMotor
    janela_inicio: date  # F2-2: dias anteriores só aquecem e não entram no teste
    cofre_inicio: date  # dias >= cofre_inicio formam o Cofre (publicado separado, veto)
    partidas_sentinela: Sequence[bool]  # True = chegou a revogado em 3 anos


@dataclass(frozen=True, slots=True)
class VereditoCandidato:
    candidato: str
    p_newey_west: float
    p_bootstrap: float
    p_newey_west_pessimista: float
    p_bootstrap_pessimista: float
    blocos_independentes: int
    p_cofre_negativo: float
    fracao_revogada: float
    nivel_aplicado: float | None
    aprovado: bool
    motivos: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VereditoPortao:
    ordem: tuple[str, ...]  # ordem de Holm (menor p primeiro)
    candidatos: tuple[VereditoCandidato, ...]
    diagnostico_deriva: Mapping[str, Any] = field(default_factory=dict)  # emenda C2, não decide
