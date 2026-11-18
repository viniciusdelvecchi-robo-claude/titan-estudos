"""Representação colunar das intenções, sem objetos por sinal.

O formato e a identidade são congelados em ``docs/contrato_intencoes_compactas.md``.
As visões de linha abaixo são transitórias e apenas expõem a interface de leitura
de :class:`Intencao`; o objeto completo só nasce por ``materializar``.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Iterable, Iterator, Sequence, overload

import numpy as np

from titan.tipos_f2 import Campanha, Intencao, Lado


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_M15_NS = 15 * 60 * 1_000_000_000
_DTYPES = (
    np.dtype(np.uint16), np.dtype(np.int8), np.dtype(np.int64),
    np.dtype(np.int64), np.dtype(np.int64), np.dtype(np.int64),
    np.dtype(np.float64),
)


@lru_cache(maxsize=262_144)
def _datetime_utc(ns: int) -> datetime:
    return _EPOCH + timedelta(microseconds=int(ns) // 1000)


@lru_cache(maxsize=262_144)
def _iso_utc(ns: int) -> str:
    return _datetime_utc(ns).isoformat()


def _digest_compacto(
    campanha: Campanha, regra_id: str, simbolo: str, lado: int,
    sinal_fim_ns: int, stop_pontos: int, hora_limite_ns: int,
    alvo_pontos: int,
) -> bytes:
    chave = "|".join((
        campanha.value,
        regra_id,
        simbolo,
        Lado.COMPRA.value if lado == 1 else Lado.VENDA.value,
        _iso_utc(sinal_fim_ns),
        str(int(stop_pontos)),
        "None" if alvo_pontos == 0 else str(int(alvo_pontos)),
        _iso_utc(hora_limite_ns),
    ))
    return hashlib.sha256(chave.encode()).digest()[:10]


def _id_compacto(
    campanha: Campanha, regra_id: str, simbolo: str, lado: int,
    sinal_fim_ns: int, stop_pontos: int, hora_limite_ns: int,
    alvo_pontos: int,
) -> str:
    return _digest_compacto(
        campanha, regra_id, simbolo, lado, sinal_fim_ns, stop_pontos,
        hora_limite_ns, alvo_pontos,
    ).hex()


@dataclass(frozen=True, slots=True, eq=False)
class LoteIntencoesCompactas(Sequence["LinhaIntencaoCompacta"]):
    campanha: Campanha
    simbolo: str
    regra_ids: tuple[str, ...]
    regra: np.ndarray
    lado: np.ndarray
    sinal_fim_ns: np.ndarray
    stop_pontos: np.ndarray
    hora_limite_ns: np.ndarray
    alvo_pontos: np.ndarray
    peso_admissao: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.campanha, Campanha):
            raise TypeError("campanha deve ser Campanha")
        if not self.simbolo:
            raise ValueError("simbolo obrigatório")
        if (
            not self.regra_ids or len(self.regra_ids) > np.iinfo(np.uint16).max + 1
            or any(not x for x in self.regra_ids)
            or len(set(self.regra_ids)) != len(self.regra_ids)
        ):
            raise ValueError("regra_ids deve ser tabela não vazia, única e válida")
        nomes = (
            "regra", "lado", "sinal_fim_ns", "stop_pontos",
            "hora_limite_ns", "alvo_pontos", "peso_admissao",
        )
        arrays = []
        tamanho = None
        for nome, dtype in zip(nomes, _DTYPES, strict=True):
            valor = getattr(self, nome)
            if not isinstance(valor, np.ndarray) or valor.dtype != dtype:
                raise TypeError(f"{nome} deve ter dtype {dtype}")
            if valor.ndim != 1 or not valor.flags.c_contiguous:
                raise ValueError(f"{nome} deve ser vetor C-contíguo")
            if tamanho is None:
                tamanho = len(valor)
            elif len(valor) != tamanho:
                raise ValueError("arrays compactos com comprimentos diferentes")
            copia = np.array(valor, dtype=dtype, order="C", copy=True)
            copia.setflags(write=False)
            object.__setattr__(self, nome, copia)
            arrays.append(copia)
        self._validar_conteudo(*arrays)

    def _validar_conteudo(
        self, regra: np.ndarray, lado: np.ndarray, sinal: np.ndarray,
        stop: np.ndarray, limite: np.ndarray, alvo: np.ndarray,
        peso: np.ndarray, *, conferir_identidade: bool = True,
    ) -> None:
        if len(regra) and int(regra.max()) >= len(self.regra_ids):
            raise ValueError("código de regra fora de regra_ids")
        if np.any((lado != 1) & (lado != -1)):
            raise ValueError("lado deve ser +1 ou -1")
        if np.any(sinal % 1000) or np.any(limite % 1000):
            raise ValueError("instantes devem ser representáveis por datetime")
        if np.any(stop <= 0) or np.any(alvo < 0):
            raise ValueError("stop deve ser positivo e alvo não negativo")
        if np.any(limite <= sinal) or np.any(limite % _M15_NS):
            raise ValueError("hora limite inválida ou fora da fronteira M15")
        if np.any(np.isnan(peso)) or np.any(peso < 0):
            raise ValueError("peso não pode ser NaN nem negativo")
        if not conferir_identidade:
            if np.any(sinal[1:] < sinal[:-1]):
                raise ValueError("lote fora da ordem canonica")
            return
        ids = np.empty(len(self), dtype="S10")
        for k in range(len(self)):
            ids[k] = _digest_compacto(
                self.campanha, self.regra_ids[int(regra[k])], self.simbolo,
                int(lado[k]), int(sinal[k]), int(stop[k]), int(limite[k]),
                int(alvo[k]),
            )
        if len(np.unique(ids)) != len(ids):
            raise ValueError("intencao_id duplicado")
        ordem = np.lexsort((ids, sinal))
        if not np.array_equal(ordem, np.arange(len(self))):
            raise ValueError("lote fora da ordem canônica")

    @classmethod
    def criar(
        cls, campanha: Campanha, simbolo: str, regra_ids: tuple[str, ...],
        regra: np.ndarray, lado: np.ndarray, sinal_fim_ns: np.ndarray,
        stop_pontos: np.ndarray, hora_limite_ns: np.ndarray,
        alvo_pontos: np.ndarray, peso_admissao: np.ndarray, *,
        canonicalizar: bool = True,
    ) -> "LoteIntencoesCompactas":
        if not isinstance(campanha, Campanha):
            raise TypeError("campanha deve ser Campanha")
        if not simbolo:
            raise ValueError("simbolo obrigatório")
        if (
            not regra_ids or len(regra_ids) > np.iinfo(np.uint16).max + 1
            or any(not x for x in regra_ids)
            or len(set(regra_ids)) != len(regra_ids)
        ):
            raise ValueError("regra_ids deve ser tabela não vazia, única e válida")
        entradas = (
            regra, lado, sinal_fim_ns, stop_pontos, hora_limite_ns,
            alvo_pontos, peso_admissao,
        )
        for nome, valor, dtype in zip(
            ("regra", "lado", "sinal_fim_ns", "stop_pontos",
             "hora_limite_ns", "alvo_pontos", "peso_admissao"),
            entradas, _DTYPES, strict=True,
        ):
            if not isinstance(valor, np.ndarray) or valor.dtype != dtype:
                raise TypeError(f"{nome} deve ter dtype {dtype}")
            if valor.ndim != 1 or not valor.flags.c_contiguous:
                raise ValueError(f"{nome} deve ser vetor C-contíguo")
        if not canonicalizar:
            n = len(regra)
            if any(len(x) != n for x in entradas):
                raise ValueError("arrays compactos com comprimentos diferentes")
            objeto = object.__new__(cls)
            object.__setattr__(objeto, "campanha", campanha)
            object.__setattr__(objeto, "simbolo", simbolo)
            object.__setattr__(objeto, "regra_ids", regra_ids)
            copias = []
            for nome, valor, dtype in zip(
                ("regra", "lado", "sinal_fim_ns", "stop_pontos",
                 "hora_limite_ns", "alvo_pontos", "peso_admissao"),
                entradas, _DTYPES, strict=True,
            ):
                copia = np.array(valor, dtype=dtype, order="C", copy=True)
                copia.setflags(write=False)
                object.__setattr__(objeto, nome, copia)
                copias.append(copia)
            objeto._validar_conteudo(*copias, conferir_identidade=False)
            return objeto
        n = len(regra)
        if any(len(x) != n for x in entradas):
            raise ValueError("arrays compactos com comprimentos diferentes")
        if n and int(regra.max()) >= len(regra_ids):
            raise ValueError("código de regra fora de regra_ids")
        ids = np.empty(n, dtype="S10")
        for k in range(n):
            ids[k] = _digest_compacto(
                campanha, regra_ids[int(regra[k])], simbolo, int(lado[k]),
                int(sinal_fim_ns[k]), int(stop_pontos[k]),
                int(hora_limite_ns[k]), int(alvo_pontos[k]),
            )
        _, primeiras = np.unique(ids, return_index=True)
        ordem_local = np.lexsort((ids[primeiras], sinal_fim_ns[primeiras]))
        ordem = np.ascontiguousarray(primeiras[ordem_local], dtype=np.int64)
        return cls(
            campanha, simbolo, regra_ids,
            *(np.ascontiguousarray(x[ordem]) for x in entradas),
        )

    def __len__(self) -> int:
        return len(self.regra)

    @overload
    def __getitem__(self, linha: int) -> "LinhaIntencaoCompacta": ...

    @overload
    def __getitem__(self, linha: slice) -> "SelecaoIntencoesCompactas": ...

    def __getitem__(self, linha):
        if isinstance(linha, slice):
            indices = np.arange(len(self), dtype=np.int64)[linha]
            return SelecaoIntencoesCompactas(self, indices)
        numero = int(linha)
        if numero < 0:
            numero += len(self)
        if numero < 0 or numero >= len(self):
            raise IndexError("linha fora do lote")
        return LinhaIntencaoCompacta(self, numero)

    def regra_id(self, linha: int) -> str:
        return self.regra_ids[int(self.regra[linha])]

    def intencao_id(self, linha: int) -> str:
        return _id_compacto(
            self.campanha, self.regra_id(linha), self.simbolo,
            int(self.lado[linha]), int(self.sinal_fim_ns[linha]),
            int(self.stop_pontos[linha]), int(self.hora_limite_ns[linha]),
            int(self.alvo_pontos[linha]),
        )

    def materializar(self, linha: int) -> Intencao:
        return self[linha].materializar()

    def selecionar(self, linhas: np.ndarray) -> "LoteIntencoesCompactas":
        if not isinstance(linhas, np.ndarray) or linhas.dtype != np.int64:
            raise TypeError("linhas deve ter dtype int64")
        if linhas.ndim != 1 or not linhas.flags.c_contiguous:
            raise ValueError("linhas deve ser vetor C-contíguo")
        if len(linhas) and (
            int(linhas[0]) < 0 or int(linhas[-1]) >= len(self)
            or np.any(linhas[1:] <= linhas[:-1])
        ):
            raise ValueError("linhas devem ser crescentes, únicas e válidas")
        return LoteIntencoesCompactas.criar(
            self.campanha, self.simbolo, self.regra_ids,
            *(np.ascontiguousarray(x[linhas]) for x in (
                self.regra, self.lado, self.sinal_fim_ns, self.stop_pontos,
                self.hora_limite_ns, self.alvo_pontos, self.peso_admissao,
            )), canonicalizar=False,
        )


class LinhaIntencaoCompacta:
    """Visão transitória e somente leitura de uma linha compacta."""

    __slots__ = ("lote", "linha", "_intencao_id")

    def __init__(self, lote: LoteIntencoesCompactas, linha: int) -> None:
        self.lote = lote
        self.linha = linha
        self._intencao_id: str | None = None

    @property
    def campanha(self) -> Campanha:
        return self.lote.campanha

    @property
    def regra_id(self) -> str:
        return self.lote.regra_id(self.linha)

    @property
    def simbolo(self) -> str:
        return self.lote.simbolo

    @property
    def lado(self) -> Lado:
        return Lado.COMPRA if self.lote.lado[self.linha] == 1 else Lado.VENDA

    @property
    def barra_sinal_fim(self) -> datetime:
        return _datetime_utc(int(self.lote.sinal_fim_ns[self.linha]))

    @property
    def stop_pontos(self) -> int:
        return int(self.lote.stop_pontos[self.linha])

    @property
    def hora_limite(self) -> datetime:
        return _datetime_utc(int(self.lote.hora_limite_ns[self.linha]))

    @property
    def alvo_pontos(self) -> int | None:
        valor = int(self.lote.alvo_pontos[self.linha])
        return None if valor == 0 else valor

    @property
    def peso_admissao(self) -> float:
        return float(self.lote.peso_admissao[self.linha])

    @property
    def intencao_id(self) -> str:
        if self._intencao_id is None:
            self._intencao_id = self.lote.intencao_id(self.linha)
        return self._intencao_id

    def materializar(self) -> Intencao:
        return Intencao(
            self.campanha, self.regra_id, self.simbolo, self.lado,
            self.barra_sinal_fim, self.stop_pontos, self.hora_limite,
            self.alvo_pontos, self.intencao_id, self.peso_admissao,
        )


@dataclass(frozen=True, slots=True)
class SelecaoIntencoesCompactas(Sequence[LinhaIntencaoCompacta]):
    """Ordem interna de linhas sem copiar as colunas do lote."""

    lote: LoteIntencoesCompactas
    indices: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.indices, np.ndarray) or self.indices.dtype != np.int64:
            raise TypeError("indices deve ter dtype int64")
        if self.indices.ndim != 1 or not self.indices.flags.c_contiguous:
            raise ValueError("indices deve ser vetor C-contíguo")
        if len(self.indices) and (
            int(self.indices.min()) < 0 or int(self.indices.max()) >= len(self.lote)
            or len(np.unique(self.indices)) != len(self.indices)
        ):
            raise ValueError("índice repetido ou fora do lote")
        copia = np.array(self.indices, dtype=np.int64, order="C", copy=True)
        copia.setflags(write=False)
        object.__setattr__(self, "indices", copia)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, numero):
        if isinstance(numero, slice):
            return SelecaoIntencoesCompactas(
                self.lote, np.ascontiguousarray(self.indices[numero]),
            )
        return self.lote[int(self.indices[numero])]


class ColecaoIntencoesCompactas(Sequence[LinhaIntencaoCompacta]):
    """Concatenação sem cópia de lotes de ativos diferentes."""

    __slots__ = ("lotes", "_fins")

    def __init__(self, lotes: Iterable[LoteIntencoesCompactas]) -> None:
        self.lotes = tuple(lotes)
        if any(not isinstance(lote, LoteIntencoesCompactas) for lote in self.lotes):
            raise TypeError("coleção aceita apenas lotes compactos")
        simbolos = [lote.simbolo for lote in self.lotes]
        if len(set(simbolos)) != len(simbolos):
            raise ValueError("a coleção aceita um único lote por ativo")
        total = 0
        fins = []
        for lote in self.lotes:
            total += len(lote)
            fins.append(total)
        self._fins = tuple(fins)

    def __len__(self) -> int:
        return self._fins[-1] if self._fins else 0

    def __getitem__(self, numero):
        if isinstance(numero, slice):
            return tuple(self)[numero]
        if numero < 0:
            numero += len(self)
        if numero < 0 or numero >= len(self):
            raise IndexError("linha fora da coleção")
        k = bisect_right(self._fins, numero)
        inicio = self._fins[k - 1] if k else 0
        return self.lotes[k][numero - inicio]

    def __iter__(self) -> Iterator[LinhaIntencaoCompacta]:
        for lote in self.lotes:
            yield from lote


SequenciaCompacta = (
    LoteIntencoesCompactas | SelecaoIntencoesCompactas
    | ColecaoIntencoesCompactas
)


def eh_sequencia_compacta(valor: object) -> bool:
    return isinstance(valor, (
        LoteIntencoesCompactas, SelecaoIntencoesCompactas,
        ColecaoIntencoesCompactas,
    ))


def ordenar_compacta(
    lote: LoteIntencoesCompactas, chave,
) -> SelecaoIntencoesCompactas:
    ordem = np.asarray(sorted(range(len(lote)), key=lambda k: chave(lote[k])), dtype=np.int64)
    return SelecaoIntencoesCompactas(lote, ordem)


__all__ = [
    "ColecaoIntencoesCompactas", "LoteIntencoesCompactas",
    "LinhaIntencaoCompacta", "SelecaoIntencoesCompactas",
    "eh_sequencia_compacta", "ordenar_compacta",
]
