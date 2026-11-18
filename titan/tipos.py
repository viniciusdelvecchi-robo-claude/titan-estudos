"""Contratos comuns da fase 1.

As datas de barras são sempre UTC com fuso explícito e os intervalos são
semiabertos: ``[inicio, fim)``. Preços e spread são inteiros em pontos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


# 09/10/2025 00:00 no relógio do servidor. Em outubro de 2025 o servidor
# estava em UTC+3 (medidas/calendario_inferido.json), logo o corte é 21:00Z.
CORTE_COFRE_UTC = datetime(2025, 10, 8, 21, tzinfo=UTC)


class TipoExecucao(StrEnum):
    EXTRATOR = "extrator"
    VALIDADOR = "validador"
    ENSAIO_CEGO = "ensaio_cego"
    ATO = "ato"
    PESQUISA = "pesquisa"
    TESTE = "teste"
    ORACULO = "oraculo"


class ModoPreco(StrEnum):
    REAL = "real"
    CEGO = "cego"
    SINTETICO = "sintetico"


class TipoFonte(StrEnum):
    ROBO_V3 = "robo_v3"
    CORRETORA = "corretora"
    SINTETICA = "sintetica"


class EstadoAptidao(StrEnum):
    APTO = "apto"
    INAPTO = "inapto"
    SUSPEITO = "suspeito"


def _utc_consciente(valor: datetime, campo: str) -> None:
    if valor.tzinfo is None or valor.utcoffset() is None:
        raise ValueError(f"{campo} deve ter fuso explícito")


@dataclass(frozen=True, slots=True)
class PedidoBarras:
    simbolo: str
    tempo_grafico: str
    inicio: datetime
    fim: datetime
    fonte_id: str
    modo: ModoPreco = ModoPreco.REAL

    def __post_init__(self) -> None:
        _utc_consciente(self.inicio, "inicio")
        _utc_consciente(self.fim, "fim")
        if self.inicio >= self.fim:
            raise ValueError("o intervalo deve satisfazer inicio < fim")
        if not self.simbolo or not self.tempo_grafico or not self.fonte_id:
            raise ValueError("simbolo, tempo_grafico e fonte_id são obrigatórios")


@dataclass(frozen=True, slots=True)
class ContextoExecucao:
    numero: int
    tipo: TipoExecucao
    finalidade: str
    calcula_resultado_regra: bool
    manifesto_hash: str
    criado_em: datetime
    semente: int | None = None

    def __post_init__(self) -> None:
        if self.numero <= 0:
            raise ValueError("a execução precisa de número positivo")
        _utc_consciente(self.criado_em, "criado_em")
        if not self.finalidade or not self.manifesto_hash:
            raise ValueError("finalidade e manifesto_hash são obrigatórios")


@dataclass(frozen=True, slots=True)
class FonteDados:
    id: str
    tipo: TipoFonte
    raiz: Path
    preco_real: bool
    somente_leitura: bool = True
    descricao: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("a fonte precisa de id")
        if self.tipo is TipoFonte.ROBO_V3 and not self.somente_leitura:
            raise ValueError("a fonte Robo V3 é obrigatoriamente somente leitura")


@dataclass(frozen=True, slots=True)
class Aptidao:
    simbolo: str
    inicio: datetime
    fim: datetime
    uso: str
    estado: EstadoAptidao
    motivos: tuple[str, ...] = ()
    evidencias: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _utc_consciente(self.inicio, "inicio")
        _utc_consciente(self.fim, "fim")
        if self.inicio >= self.fim:
            raise ValueError("período de aptidão inválido")
        if self.estado is EstadoAptidao.INAPTO and not self.motivos:
            raise ValueError("inaptidão precisa de motivo")


@dataclass(frozen=True, slots=True)
class Sessao:
    simbolo: str
    data_servidor: date
    abertura_utc: datetime | None
    fechamento_utc: datetime | None
    limite_zeragem_utc: datetime | None
    virada_utc: datetime
    offset_servidor_min: int
    feriado: bool
    fonte: str
    evidencia: str = ""

    def __post_init__(self) -> None:
        _utc_consciente(self.virada_utc, "virada_utc")
        for nome in ("abertura_utc", "fechamento_utc", "limite_zeragem_utc"):
            valor = getattr(self, nome)
            if valor is not None:
                _utc_consciente(valor, nome)
        if self.feriado:
            if any((self.abertura_utc, self.fechamento_utc, self.limite_zeragem_utc)):
                raise ValueError("feriado não pode declarar janela de negociação")
        elif not all((self.abertura_utc, self.fechamento_utc, self.limite_zeragem_utc)):
            raise ValueError("sessão negociável precisa de abertura, fechamento e limite")
        elif not (self.abertura_utc < self.limite_zeragem_utc <= self.fechamento_utc):
            raise ValueError("ordem temporal da sessão inválida")
        if not self.fonte:
            raise ValueError("a sessão precisa declarar a fonte")


@dataclass(frozen=True, slots=True)
class ManifestoColeta:
    data_servidor: date
    criado_em: datetime
    fonte_id: str
    arquivos: tuple[Mapping[str, Any], ...]
    completo: bool
    erros: tuple[str, ...] = ()
    metadados: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _utc_consciente(self.criado_em, "criado_em")
        if not self.fonte_id:
            raise ValueError("fonte_id é obrigatório")
        if self.completo and self.erros:
            raise ValueError("coleta completa não pode conter erros")

