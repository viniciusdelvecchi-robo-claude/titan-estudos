"""Motor deterministico da fase 2 (F2-1b, F2-3--F2-6 e F2-13--F2-15).

O modulo nao abre uma segunda familia de contratos comuns: ``Intencao``,
``ParamsMotor`` e ``ResultadoMotor`` vem de :mod:`titan.tipos_f2`.  As tres
dataclasses abaixo descrevem somente a entrada economica do simulador e sao
deliberadamente pequenas para que dados sinteticos e o carregador real usem a
mesma porta.

Precos e spreads sao pontos inteiros.  A barra usa preco bid; compra entra no
ask (bid + spread) e venda entra no bid. O valor do ponto fica na moeda de
cotacao e cada execucao usa o cambio USD da barra; nao ha dupla conversao.
Sem caminho intrabar declarado, stop precede alvo quando ambos cabem na barra
(F2-3b).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
import heapq
from math import floor
from statistics import median
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence, TypeVar

import numpy as np

from titan.intencoes_compactas import (
    ColecaoIntencoesCompactas,
    LoteIntencoesCompactas,
    SelecaoIntencoesCompactas,
    _datetime_utc,
    eh_sequencia_compacta,
)
from titan.tipos import Sessao, _utc_consciente
from titan.tipos_f2 import (
    Campanha,
    EstadoRisco,
    Intencao,
    Lado,
    MotivoRecusa,
    ParamsMotor,
    Recusa,
    ResultadoMotor,
)


@dataclass(frozen=True, slots=True)
class BarraMotor:
    """Uma barra negociavel entregue ao motor.

    ``caminho_bid`` e opcional e, quando presente, e a sequencia observada
    depois da abertura (por exemplo, barras de um minuto dentro da M15). Sem
    ela, o motor aplica o caminho conservador.
    """

    simbolo: str
    inicio: datetime
    tempo_grafico: str
    abertura: int
    maxima: int
    minima: int
    fechamento: int
    spread: int
    data_servidor: date
    caminho_bid: tuple[int, ...] = ()
    cambio_usd: float = 1.0
    apto: bool = True

    def __post_init__(self) -> None:
        _utc_consciente(self.inicio, "inicio")
        if not self.simbolo or self.tempo_grafico not in {"M15", "H1", "D1"}:
            raise ValueError("simbolo e tempo grafico M15/H1/D1 sao obrigatorios")
        if self.spread < 0 or self.cambio_usd <= 0:
            raise ValueError("spread nao negativo e cambio positivo")
        if not self.minima <= min(self.abertura, self.fechamento) <= max(
            self.abertura, self.fechamento
        ) <= self.maxima:
            raise ValueError("OHLC inconsistente")
        if any(p < self.minima or p > self.maxima for p in self.caminho_bid):
            raise ValueError("caminho fora da faixa da barra")

    @property
    def movimento(self) -> int:
        return self.maxima - self.minima


@dataclass(frozen=True, slots=True)
class AjusteMotor:
    """Dividendo ou swap por lote, convertido pela taxa da propria data."""

    instante: datetime
    tipo: str  # "dividendo" ou "swap"
    valor_por_lote: float
    cambio_usd: float = 1.0

    def __post_init__(self) -> None:
        _utc_consciente(self.instante, "instante")
        if self.tipo not in {"dividendo", "swap"} or self.cambio_usd <= 0:
            raise ValueError("ajuste invalido")


@dataclass(frozen=True, slots=True)
class EspecificacaoAtivo:
    """Metadados congelados do contrato usados em tamanho, margem e custos.

    ``valor_ponto_moeda_cotacao`` nao esta em USD: cada execucao o converte
    com ``BarraMotor.cambio_usd``. ``stop_out_nivel_margem`` usa a definicao
    da corretora, patrimonio/margem, e deve vir do cadastro do contrato.
    """

    simbolo: str
    classe: str
    valor_ponto_moeda_cotacao: float
    lote_minimo: float
    passo_lote: float
    margem_por_lote_usd: float | None
    stop_out_nivel_margem: float
    taxa_margem_notional: float | None = None
    slippage_pontos: int = 0
    stop_minimo_pontos: int = 1
    modo_contrato: str = "spot"
    ajustes: tuple[AjusteMotor, ...] = ()
    mudancas_margem: tuple[tuple[datetime, float], ...] = ()
    custo_convencao_diaria_usd_por_lote: float = 0.0
    dividendo_pessimista_usd_por_lote_dia: float = 0.0

    def __post_init__(self) -> None:
        if not self.simbolo or not self.classe:
            raise ValueError("simbolo e classe obrigatorios")
        if min(
            self.valor_ponto_moeda_cotacao,
            self.lote_minimo,
            self.passo_lote,
            self.stop_out_nivel_margem,
        ) <= 0:
            raise ValueError("valores economicos devem ser positivos")
        margem_fixa = self.margem_por_lote_usd is not None
        margem_taxa = self.taxa_margem_notional is not None
        if margem_fixa == margem_taxa:
            raise ValueError("informe exatamente uma margem: USD por lote ou taxa notional")
        if margem_fixa and self.margem_por_lote_usd <= 0:
            raise ValueError("margem por lote deve ser positiva")
        if margem_taxa and self.taxa_margem_notional <= 0:
            raise ValueError("taxa de margem notional deve ser positiva")
        if margem_taxa and self.mudancas_margem:
            raise ValueError("mudancas_margem em USD não cabem no modo de taxa notional")
        if self.slippage_pontos < 0 or self.stop_minimo_pontos <= 0:
            raise ValueError("slippage nao negativo e stop minimo positivo")
        if self.dividendo_pessimista_usd_por_lote_dia < 0:
            raise ValueError("custo pessimista de dividendo nao pode ser negativo")
        anterior: datetime | None = None
        for instante, valor in self.mudancas_margem:
            _utc_consciente(instante, "mudanca de margem")
            if valor <= 0 or (anterior is not None and instante <= anterior):
                raise ValueError("mudancas de margem devem ser positivas e ordenadas")
            anterior = instante

    @property
    def limita_um_terco(self) -> bool:
        return self.classe.lower() in {"indice", "indice_spot", "acao", "acoes"}

    def margem_no_instante(
        self,
        instante: datetime,
        preco_pontos: int | None = None,
        cambio_usd: float | None = None,
    ) -> float:
        if self.taxa_margem_notional is None:
            valor = self.margem_por_lote_usd
            for quando, novo in self.mudancas_margem:
                if quando > instante:
                    break
                valor = novo
            return valor
        if preco_pontos is None or cambio_usd is None:
            raise ValueError("margem por taxa exige preço em pontos e câmbio USD")
        if preco_pontos <= 0 or cambio_usd <= 0:
            raise ValueError("preço e câmbio da margem devem ser positivos")
        return (
            self.taxa_margem_notional
            * preco_pontos
            * self.valor_ponto_moeda_cotacao
            * cambio_usd
        )


@dataclass(slots=True)
class _Posicao:
    intencao: Intencao
    especificacao: EspecificacaoAtivo
    quantidade: float
    entrada_em: datetime
    entrada: int
    stop: int
    alvo: int | None
    spread_entrada: int
    risco_entrada_usd: float
    margem_entrada_usd: float
    bloco: str
    fallback_diario: bool = False
    ajustes_usd: float = 0.0
    ultimo_bid: int = 0
    ultimo_spread: int = 0
    ultimo_spread_real: int = 0
    ultimo_cambio_usd: float = 1.0


def _margem_posicao_por_lote(posicao: _Posicao, instante: datetime) -> float:
    return posicao.especificacao.margem_no_instante(
        instante, posicao.ultimo_bid, posicao.ultimo_cambio_usd
    )


def _ceil_m15(instante: datetime) -> datetime:
    base = instante.replace(second=0, microsecond=0)
    resto = base.minute % 15
    if resto or instante.second or instante.microsecond:
        base += timedelta(minutes=15 - resto if resto else 15)
    return base


def _com_hora_limite(intencao: Any, limite: datetime) -> Any:
    """Materializa uma linha compacta só se a posição exige limite efetivo."""
    materializar = getattr(intencao, "materializar", None)
    if intencao.hora_limite == limite:
        return materializar() if materializar is not None else intencao
    if materializar is not None:
        intencao = materializar()
    return replace(intencao, hora_limite=limite)


def _barras_do(carregador: Any, simbolo: str) -> tuple[BarraMotor, ...]:
    if isinstance(carregador, Mapping):
        bruto = carregador.get(simbolo, ())
    elif hasattr(carregador, "barras_motor"):
        bruto = carregador.barras_motor(simbolo)
    else:
        bruto = carregador.barras(simbolo)
    barras = tuple(bruto)
    ordenadas = True
    chave_anterior: tuple[datetime, str] | None = None
    for b in barras:
        if not isinstance(b, BarraMotor) or b.simbolo != simbolo:
            raise TypeError("carregador deve entregar BarraMotor do simbolo pedido")
        chave = (b.inicio, b.tempo_grafico)
        if chave_anterior is not None and chave < chave_anterior:
            ordenadas = False
        chave_anterior = chave
    return barras if ordenadas else tuple(sorted(barras, key=lambda b: (b.inicio, b.tempo_grafico)))


def _especificacao(contratos: Any, simbolo: str, instante: datetime) -> EspecificacaoAtivo:
    if isinstance(contratos, Mapping):
        e = contratos[simbolo]
    elif hasattr(contratos, "especificacao"):
        e = contratos.especificacao(simbolo, instante)
    else:
        e = contratos(simbolo, instante)
    if not isinstance(e, EspecificacaoAtivo) or e.simbolo != simbolo:
        raise TypeError("contrato deve entregar EspecificacaoAtivo do simbolo pedido")
    return e


def _sessao(calendario: Any, simbolo: str, dia: date) -> Sessao:
    if callable(calendario):
        s = calendario(simbolo, dia)
    elif (simbolo, dia) in calendario:
        s = calendario[(simbolo, dia)]
    elif simbolo in calendario and isinstance(calendario[simbolo], Mapping):
        s = calendario[simbolo][dia]
    else:
        raise KeyError((simbolo, dia))
    if not isinstance(s, Sessao) or s.simbolo != simbolo or s.data_servidor != dia:
        raise TypeError("calendario deve entregar Sessao correspondente")
    return s


def _sessoes_enumeraveis(
    calendario: Any, simbolo: str
) -> dict[date, Sessao]:
    """Materializa as sessões declaradas para detectar ausência no Cofre.

    Um callable comum é o método ligado ``mundo.calendario``; nesse caso o
    catálogo está em ``mundo.sessoes``. Sem uma fonte enumerável, ausência de
    barra não é distinguível de ausência de pregão e o motor falha fechado.
    """
    fonte = calendario
    dono = getattr(calendario, "__self__", None)
    if dono is not None and hasattr(dono, "sessoes"):
        fonte = dono.sessoes
    elif hasattr(calendario, "sessoes"):
        fonte = calendario.sessoes
    if isinstance(fonte, Mapping):
        if simbolo in fonte and isinstance(fonte[simbolo], Mapping):
            itens = fonte[simbolo].items()
        else:
            itens = (
                (chave[1], valor)
                for chave, valor in fonte.items()
                if isinstance(chave, tuple) and len(chave) == 2
                and chave[0] == simbolo
            )
        saida = {
            d: s for d, s in itens
            if isinstance(d, date) and isinstance(s, Sessao)
        }
        if saida:
            return saida
    raise ValueError(
        "cofre_inicio exige calendario enumeravel por simbolo e data"
    )


def _pior_spread_hora_antes(
    barras: Sequence[BarraMotor], calendario: Any, sessao_ausente: Sessao,
    janela: int,
) -> int | None:
    """Maior spread da hora nos últimos ``janela`` pregões observados."""
    hora = (
        sessao_ausente.abertura_utc
        + timedelta(minutes=sessao_ausente.offset_servidor_min)
    ).hour % 24
    maximo_dia: dict[date, int] = {}
    for b in barras:
        if (
            b.tempo_grafico != "M15"
            or b.data_servidor >= sessao_ausente.data_servidor
            or _hora_servidor(b, calendario) != hora
        ):
            continue
        maximo_dia[b.data_servidor] = max(
            maximo_dia.get(b.data_servidor, b.spread), b.spread
        )
    dias = sorted(maximo_dia)[-janela:]
    return max((maximo_dia[d] for d in dias), default=None)


def _passo_quantidade(valor: float, spec: EspecificacaoAtivo) -> float:
    passos = floor((valor + 1e-12) / spec.passo_lote)
    return round(passos * spec.passo_lote, 10)


def _quantidade(
    valor: float, spec: EspecificacaoAtivo, lote_continuo_diagnostico: bool
) -> float:
    """Dimensiona normalmente ou, só em D4.9, sem mínimo nem passo."""
    return valor if lote_continuo_diagnostico else _passo_quantidade(valor, spec)


def _preco_entrada(intencao: Intencao, barra: BarraMotor, spread: int,
                   spec: EspecificacaoAtivo) -> int:
    """Preco executado, sempre com derrapagem adversa."""
    if intencao.lado is Lado.COMPRA:
        return barra.abertura + spread + spec.slippage_pontos
    return barra.abertura - spec.slippage_pontos


def _preco_saida(pos: _Posicao, bid: int, spread: int, *, slippage: bool = False,
                 spread_real: int | None = None) -> int:
    derrapagem = pos.especificacao.slippage_pontos if slippage else 0
    spread_real = spread if spread_real is None else spread_real
    piora_spread = max(0, spread - spread_real)
    if pos.intencao.lado is Lado.COMPRA:
        # OHLC e bid; no pessimista, alarga-se o lado de saida pela diferenca
        # entre o pior spread historico e o spread observado.
        return bid - piora_spread - derrapagem
    return bid + spread + derrapagem


def _pnl(pos: _Posicao, saida: int, cambio: float) -> float:
    pontos = saida - pos.entrada if pos.intencao.lado is Lado.COMPRA else pos.entrada - saida
    return (pontos * pos.especificacao.valor_ponto_moeda_cotacao
            * pos.quantidade * cambio + pos.ajustes_usd)


def _risco_atual(pos: _Posicao, bid: int, spread: int, cambio: float) -> float:
    atual = _preco_saida(pos, bid, spread, spread_real=pos.ultimo_spread_real)
    pontos = atual - pos.stop if pos.intencao.lado is Lado.COMPRA else pos.stop - atual
    return max(0.0, pontos * pos.especificacao.valor_ponto_moeda_cotacao
               * pos.quantidade * cambio)


def _atingimento(pos: _Posicao, barra: BarraMotor, *, spread_real: int) -> tuple[str, int] | None:
    """Devolve o primeiro gatilho; sem caminho conhecido, stop vence alvo."""
    if pos.intencao.lado is Lado.COMPRA:
        if barra.abertura <= pos.stop:
            return "stop_gap", _preco_saida(
                pos, barra.abertura, barra.spread, slippage=True, spread_real=spread_real
            )
        stop_cabe = barra.minima <= pos.stop
        alvo_cabe = pos.alvo is not None and barra.maxima >= pos.alvo
    else:
        if barra.abertura + barra.spread >= pos.stop:
            return "stop_gap", _preco_saida(
                pos, barra.abertura, barra.spread, slippage=True, spread_real=spread_real
            )
        # O OHLC e bid. A compra de encerramento paga spread.
        stop_cabe = barra.maxima + barra.spread >= pos.stop
        alvo_cabe = pos.alvo is not None and barra.minima + barra.spread <= pos.alvo
    if barra.caminho_bid:
        for bid in barra.caminho_bid:
            ask = bid + barra.spread
            if pos.intencao.lado is Lado.COMPRA:
                if bid <= pos.stop:
                    return "stop", pos.stop - pos.especificacao.slippage_pontos - max(
                        0, barra.spread - spread_real
                    )
                if pos.alvo is not None and bid >= pos.alvo:
                    return "alvo", pos.alvo
            else:
                if ask >= pos.stop:
                    return "stop", pos.stop + pos.especificacao.slippage_pontos + max(
                        0, barra.spread - spread_real
                    )
                if pos.alvo is not None and ask <= pos.alvo:
                    return "alvo", pos.alvo
        return None
    if stop_cabe:
        delta = pos.especificacao.slippage_pontos + max(0, barra.spread - spread_real)
        return "stop", pos.stop - delta if pos.intencao.lado is Lado.COMPRA else pos.stop + delta
    if alvo_cabe:
        return "alvo", pos.alvo  # type: ignore[return-value]
    return None


def _hora_servidor(barra: BarraMotor, calendario: Any) -> int:
    try:
        s = _sessao(calendario, barra.simbolo, barra.data_servidor)
    except (KeyError, TypeError):
        return barra.inicio.hour
    return (barra.inicio + timedelta(minutes=s.offset_servidor_min)).hour % 24


@dataclass(frozen=True, slots=True)
class _IndicadorHora:
    observacoes: int
    spread_mediano: float | None
    movimento_mediano: float | None
    spread_maximo: int | None


def _remover_ordenado(valores: list[float], valor: float) -> None:
    indice = bisect_left(valores, valor)
    if indice >= len(valores) or valores[indice] != valor:
        raise RuntimeError("janela causal de hora inconsistente")
    valores.pop(indice)


def _precalcular_horas(
    barras_por_simbolo: Mapping[str, Sequence[BarraMotor]], calendario: Any, janela: int,
    chaves_necessarias: set[tuple[str, str, int]] | None = None,
) -> dict[tuple[str, str, int, date], _IndicadorHora]:
    """Uma passagem causal; consultas posteriores sao O(1) (R4)."""
    saida: dict[tuple[str, str, int, date], _IndicadorHora] = {}
    for simbolo, barras in barras_por_simbolo.items():
        offsets: dict[date, int] = {}
        for dia in {b.data_servidor for b in barras}:
            try:
                offsets[dia] = _sessao(calendario, simbolo, dia).offset_servidor_min
            except (KeyError, TypeError):
                offsets[dia] = 0
        grupos: dict[tuple[str, int], dict[date, list[BarraMotor]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for b in barras:
            hora = (b.inicio + timedelta(minutes=offsets[b.data_servidor])).hour % 24
            if chaves_necessarias is not None and (simbolo, b.tempo_grafico, hora) not in chaves_necessarias:
                continue
            grupos[(b.tempo_grafico, hora)][b.data_servidor].append(b)
        for (tempo_grafico, hora), por_dia in grupos.items():
            spreads_ord: list[float] = []
            movimentos_ord: list[float] = []
            janela_dias: deque[tuple[int, float, float, int]] = deque()
            maximos: deque[tuple[int, int]] = deque()
            for indice, dia in enumerate(sorted(por_dia)):
                n = len(janela_dias)
                saida[(simbolo, tempo_grafico, hora, dia)] = _IndicadorHora(
                    n,
                    float(median(spreads_ord)) if n >= janela else None,
                    float(median(movimentos_ord)) if n >= janela else None,
                    maximos[0][1] if n >= janela else None,
                )
                bs = por_dia[dia]
                spread_dia = float(median(b.spread for b in bs))
                movimento_dia = float(max(b.maxima for b in bs) - min(b.minima for b in bs))
                maximo_dia = max(b.spread for b in bs)
                insort(spreads_ord, spread_dia)
                insort(movimentos_ord, movimento_dia)
                janela_dias.append((indice, spread_dia, movimento_dia, maximo_dia))
                while maximos and maximos[-1][1] <= maximo_dia:
                    maximos.pop()
                maximos.append((indice, maximo_dia))
                if len(janela_dias) > janela:
                    antigo_i, antigo_s, antigo_m, _ = janela_dias.popleft()
                    _remover_ordenado(spreads_ord, antigo_s)
                    _remover_ordenado(movimentos_ord, antigo_m)
                    if maximos and maximos[0][0] == antigo_i:
                        maximos.popleft()
    return saida


def _indicador_hora(
    cache: Mapping[tuple[str, str, int, date], _IndicadorHora], barra: BarraMotor,
    calendario: Any,
) -> _IndicadorHora:
    return cache.get(
        (barra.simbolo, barra.tempo_grafico, _hora_servidor(barra, calendario), barra.data_servidor),
        _IndicadorHora(0, None, None, None),
    )


def _spread_execucao(
    cache: Mapping[tuple[str, str, int, date], _IndicadorHora], barra: BarraMotor,
    params: ParamsMotor, calendario: Any,
) -> int:
    if not params.pessimista:
        return barra.spread
    indicador = _indicador_hora(cache, barra, calendario)
    return indicador.spread_maximo if indicador.spread_maximo is not None else barra.spread


@dataclass(frozen=True, slots=True)
class _IndiceBarras:
    m15_por_inicio: Mapping[datetime, BarraMotor]
    intradia_tempos: tuple[datetime, ...]
    intradia: tuple[BarraMotor, ...]
    diaria_datas: tuple[date, ...]
    diarias: tuple[BarraMotor, ...]


@dataclass(frozen=True, slots=True)
class _PrecalculoCompartilhado:
    barras_por_simbolo: Mapping[str, tuple[BarraMotor, ...]]
    indices_barras: Mapping[str, _IndiceBarras]
    barras_evento: Mapping[datetime, tuple[BarraMotor, ...]]
    indicadores_hora: Mapping[tuple[str, str, int, date], _IndicadorHora]
    primeiro_instante: Mapping[str, datetime]
    datas: Mapping[str, frozenset[date]]
    indicadores_por_barra: dict[int, _IndicadorHora] = field(
        default_factory=dict, compare=False,
    )


class _ListaDescartavel(list):
    """Coletor nulo usado quando o estudo pede somente as series economicas."""

    def append(self, item: Any) -> None:
        pass


@dataclass(frozen=True, slots=True)
class _AdmissoesPreparadas:
    filas: Mapping[str, Any]

    def __iter__(self):
        for simbolo in sorted(self.filas):
            yield from self.filas[simbolo]


@dataclass(frozen=True, slots=True)
class _FilaCompacta:
    """Linhas de um lote na prioridade da barra, materializadas sob demanda."""

    lote: LoteIntencoesCompactas
    linhas: np.ndarray
    barra: BarraMotor
    cache: dict[int, Any] = field(default_factory=dict, compare=False)

    def __len__(self) -> int:
        return len(self.linhas)

    def __getitem__(self, numero: int) -> tuple[Intencao, BarraMotor, bool]:
        linha = int(self.linhas[numero])
        intencao = self.cache.get(linha)
        if intencao is None:
            intencao = self.lote[linha]
            self.cache[linha] = intencao
        return intencao, self.barra, False


def _preparar_admissoes(
    itens: Sequence[tuple[Intencao, BarraMotor, bool]],
) -> _AdmissoesPreparadas:
    grupos: defaultdict[str, list[tuple[Intencao, BarraMotor, bool]]] = defaultdict(list)
    for item in itens:
        grupos[item[0].simbolo].append(item)
    return _AdmissoesPreparadas({
        simbolo: tuple(sorted(
            grupo, key=lambda x: (-x[0].peso_admissao, x[0].intencao_id),
        ))
        for simbolo, grupo in grupos.items()
    })


def _admissoes_enxutas(
    itens: Sequence[tuple[Intencao, BarraMotor, bool]] | _AdmissoesPreparadas,
    simbolos_bloqueados: set[str],
):
    """Mescla prioridades por símbolo e abandona a fila após uma admissão.

    O conjunto é compartilhado com o laço chamador: quando ele acrescenta um
    símbolo aceito, a próxima cabeça desse símbolo é descartada e nenhuma outra
    intenção da fila é visitada.
    """
    if isinstance(itens, _AdmissoesPreparadas):
        filas = {
            simbolo: fila for simbolo, fila in itens.filas.items()
            if simbolo not in simbolos_bloqueados
        }
    else:
        grupos: defaultdict[str, list[tuple[Intencao, BarraMotor, bool]]] = defaultdict(list)
        for item in itens:
            if item[0].simbolo not in simbolos_bloqueados:
                grupos[item[0].simbolo].append(item)
        filas = {
            simbolo: sorted(
                grupo, key=lambda x: (-x[0].peso_admissao, x[0].intencao_id)
            )
            for simbolo, grupo in grupos.items()
        }
    heap = []
    for simbolo, fila in filas.items():
        if fila:
            i = fila[0][0]
            heapq.heappush(
                heap, ((-i.peso_admissao, i.intencao_id), simbolo, 0, fila[0])
            )
    while heap:
        _, simbolo, numero, item = heapq.heappop(heap)
        if simbolo in simbolos_bloqueados:
            continue
        yield item
        if simbolo in simbolos_bloqueados:
            continue
        seguinte = numero + 1
        if seguinte < len(filas[simbolo]):
            proximo = filas[simbolo][seguinte]
            i = proximo[0]
            heapq.heappush(
                heap,
                ((-i.peso_admissao, i.intencao_id), simbolo, seguinte, proximo),
            )


def _indexar_barras(barras: Sequence[BarraMotor]) -> _IndiceBarras:
    """Índice de admissão; barras ``apto=False`` ficam fora só da entrada.

    O relógio e a marcação usam ``barras_por_simbolo`` bruto. Assim, sair do
    universo impede nova ordem sem apagar preços necessários ao stop/L de uma
    posição que já estava aberta (D0a).
    """
    m15: dict[datetime, BarraMotor] = {}
    intradia_lista: list[BarraMotor] = []
    diarias_lista: list[BarraMotor] = []
    for b in barras:
        if not b.apto:
            continue
        if b.tempo_grafico == "M15":
            m15[b.inicio] = b
            intradia_lista.append(b)
        elif b.tempo_grafico == "H1":
            intradia_lista.append(b)
        else:
            diarias_lista.append(b)
    intradia = tuple(intradia_lista)
    diarias = tuple(diarias_lista)
    return _IndiceBarras(
        m15, tuple(b.inicio for b in intradia), intradia,
        tuple(b.data_servidor for b in diarias), diarias,
    )


def _selecionar_entrada(
    intencao: Intencao, indice: _IndiceBarras
) -> tuple[BarraMotor | None, bool]:
    if intencao.campanha is Campanha.DAY_TRADE_M15:
        instante = _ceil_m15(intencao.barra_sinal_fim)
        return indice.m15_por_inicio.get(instante), False
    # Campanha diaria: primeiro preco intradiario negociavel depois da janela.
    pos = bisect_left(indice.intradia_tempos, intencao.barra_sinal_fim)
    if pos < len(indice.intradia):
        return indice.intradia[pos], False
    pos = bisect_left(indice.diaria_datas, intencao.barra_sinal_fim.date() + timedelta(days=1))
    diaria = indice.diarias[pos] if pos < len(indice.diarias) else None
    return diaria, diaria is not None


def simular(
    intencoes_por_barra: Mapping[datetime, Sequence[Intencao]] | Iterable[Intencao],
    carregador: Any,
    contratos: Any,
    calendario: Any,
    params: ParamsMotor,
    *,
    somente_series: bool = False,
    cofre_inicio: date | None = None,
    lote_continuo_diagnostico: bool = False,
    alarme: Callable[[Mapping[str, Any]], None] | Any | None = None,
    _otimizar: bool = True,
    _precalculo: _PrecalculoCompartilhado | None = None,
    _somente_series: bool | None = None,
    _eventos_preparados: Mapping[
        datetime, Sequence[tuple[Intencao, BarraMotor, bool]]
    ] | None = None,
    _intencoes_validadas: bool = False,
    _simbolos_preparados: Sequence[str] | None = None,
) -> ResultadoMotor:
    """Simula o motor com tamanho diario a capital constante de US$ 3.000.

    Com ``params.capital_constante=True``, o patrimonio acumulado e mantido
    apenas para os quatro estados da secao 4 e nunca entra no tamanho posterior
    (F2-1b). Com ``False``, o mesmo caminho executa as partidas do sistema
    completo de F2-10, dimensionando pelo patrimonio do inicio do dia.
    ``cofre_inicio`` ativa D0(b): sessão aberta do Cofre sem nenhuma M15 bruta
    fecha no primeiro dia faltante qualquer posição aberta do símbolo.
    ``lote_continuo_diagnostico=True`` é exclusivamente D4.9: usa a quantidade
    fracionária exata, sem lote mínimo nem passo. Risco, margem e todos os
    demais controles continuam idênticos; o padrão ``False`` é o ato normal.
    """
    if _somente_series is None:
        _somente_series = somente_series
    if isinstance(intencoes_por_barra, Mapping):
        intencoes = [i for instante in sorted(intencoes_por_barra) for i in intencoes_por_barra[instante]]
    elif eh_sequencia_compacta(intencoes_por_barra):
        intencoes = intencoes_por_barra
    elif _somente_series and isinstance(intencoes_por_barra, list):
        intencoes = intencoes_por_barra
    else:
        intencoes = list(intencoes_por_barra)
    if (
        not _intencoes_validadas
        and not eh_sequencia_compacta(intencoes)
        and len({i.intencao_id for i in intencoes}) != len(intencoes)
    ):
        raise ValueError("intencao_id duplicado")

    simbolos = (
        list(_simbolos_preparados)
        if _simbolos_preparados is not None
        else sorted({i.simbolo for i in intencoes})
    )
    if _precalculo is None:
        barras_por_simbolo = {s: _barras_do(carregador, s) for s in simbolos}
        indices_barras = {s: _indexar_barras(b) for s, b in barras_por_simbolo.items()}
    else:
        faltantes = set(simbolos) - set(_precalculo.barras_por_simbolo)
        if faltantes:
            raise ValueError(f"pre-calculo sem simbolos: {sorted(faltantes)}")
        barras_por_simbolo = {s: _precalculo.barras_por_simbolo[s] for s in simbolos}
        indices_barras = {s: _precalculo.indices_barras[s] for s in simbolos}
    eventos: Mapping[datetime, Sequence[tuple[Intencao, BarraMotor, bool]]]
    eventos = _eventos_preparados if _eventos_preparados is not None else defaultdict(list)
    recusas: list[Recusa] = _ListaDescartavel() if _somente_series else []

    if _eventos_preparados is None:
        ordem_intencoes = (
            intencoes if _somente_series
            else sorted(intencoes, key=lambda x: (x.barra_sinal_fim, x.intencao_id))
        )
        for i in ordem_intencoes:
            barra, fallback = _selecionar_entrada(i, indices_barras[i.simbolo])
            entrada_prevista = _ceil_m15(i.barra_sinal_fim) if i.campanha is Campanha.DAY_TRADE_M15 else i.barra_sinal_fim
            if entrada_prevista >= i.hora_limite:
                if not _somente_series:
                    recusas.append(Recusa(i.intencao_id, i.barra_sinal_fim, MotivoRecusa.ZERAGEM, {
                        "entrada_prevista": entrada_prevista.isoformat(), "hora_limite": i.hora_limite.isoformat(),
                        "relacao": "entrada_prevista >= hora_limite",
                    }))
            elif barra is None:
                if not _somente_series:
                    recusas.append(Recusa(i.intencao_id, entrada_prevista, MotivoRecusa.INAPTO, {
                        "tempo_grafico_exigido": "M15" if i.campanha is Campanha.DAY_TRADE_M15 else "H1_ou_D1_seguinte",
                        "relacao": "barra_negociavel_ausente",
                    }))
            elif barra.inicio >= i.hora_limite:
                if not _somente_series:
                    recusas.append(Recusa(i.intencao_id, barra.inicio, MotivoRecusa.ZERAGEM, {
                        "entrada": barra.inicio.isoformat(), "hora_limite": i.hora_limite.isoformat(),
                        "relacao": "entrada >= hora_limite",
                    }))
            else:
                eventos[barra.inicio].append((i, barra, fallback))  # type: ignore[attr-defined]

    # O relogio do motor e a uniao das barras, inclusive as sem intencao.
    if _precalculo is None:
        barras_evento: Mapping[datetime, Sequence[BarraMotor]] = defaultdict(list)
        for barras in barras_por_simbolo.values():
            dias_com_intradia = {
                b.data_servidor for b in barras if b.tempo_grafico in {"M15", "H1"}
            }
            for b in barras:
                # Evita processar D1 e intradia simultaneamente quando ha detalhe.
                if b.tempo_grafico != "D1" or b.data_servidor not in dias_com_intradia:
                    barras_evento[b.inicio].append(b)
    else:
        barras_evento = _precalculo.barras_evento

    faltas_cofre: defaultdict[
        datetime, list[tuple[str, date, Sessao]]
    ] = defaultdict(list)
    spreads_falta: dict[tuple[str, date], int | None] = {}
    if cofre_inicio is not None:
        for simbolo in simbolos:
            dias_m15 = {
                b.data_servidor for b in barras_por_simbolo[simbolo]
                if b.tempo_grafico == "M15"
            }
            for dia, s in sorted(
                _sessoes_enumeraveis(calendario, simbolo).items()
            ):
                if (
                    dia < cofre_inicio or dia in dias_m15 or s.feriado
                    or s.abertura_utc is None or s.fechamento_utc is None
                ):
                    continue
                faltas_cofre[s.abertura_utc].append((simbolo, dia, s))
                spreads_falta[(simbolo, dia)] = _pior_spread_hora_antes(
                    barras_por_simbolo[simbolo], calendario, s,
                    params.filtro_janela_pregoes,
                )

    horas_entrada = (
        set()
        if _precalculo is not None
        else {
            (b.simbolo, b.tempo_grafico, _hora_servidor(b, calendario))
            for itens in eventos.values() for _, b, _ in itens
        }
    )
    indicadores_hora = (
        _precalcular_horas(
            barras_por_simbolo,
            calendario,
            params.filtro_janela_pregoes,
            None if params.pessimista else horas_entrada,
        )
        if _precalculo is None else _precalculo.indicadores_hora
    )
    indicadores_por_barra = (
        {} if _precalculo is None else _precalculo.indicadores_por_barra
    )

    def indicador_barra(barra: BarraMotor) -> _IndicadorHora:
        chave = id(barra)
        indicador = indicadores_por_barra.get(chave)
        if indicador is None:
            indicador = _indicador_hora(indicadores_hora, barra, calendario)
            indicadores_por_barra[chave] = indicador
        return indicador

    def spread_barra(barra: BarraMotor) -> int:
        if not params.pessimista:
            return barra.spread
        indicador = indicador_barra(barra)
        return (
            indicador.spread_maximo
            if indicador.spread_maximo is not None else barra.spread
        )

    specs = {
        s: _especificacao(
            contratos, s,
            _precalculo.primeiro_instante.get(s, datetime.now(UTC))
            if _precalculo is not None
            else min(
                (b.inicio for b in barras_por_simbolo[s]),
                default=datetime.now(UTC),
            ),
        )
        for s in simbolos
    }
    cambio_constante = all(
        not bs or all(b.cambio_usd == bs[0].cambio_usd for b in bs)
        for bs in barras_por_simbolo.values()
    )
    controles_implicitos = (
        _otimizar
        and _somente_series
        and params.capital_constante
        and cofre_inicio is None
        and not lote_continuo_diagnostico
        and getattr(carregador, "exposicao_conhecida", None) is None
        and cambio_constante
        and all(
            spec.taxa_margem_notional is None
            and not spec.mudancas_margem
            and not spec.ajustes
            and not spec.custo_convencao_diaria_usd_por_lote
            and not spec.dividendo_pessimista_usd_por_lote_dia
            for spec in specs.values()
        )
    )
    posicoes: list[_Posicao] = []
    livro: list[dict[str, Any]] = _ListaDescartavel() if _somente_series else []
    estados: list[tuple[datetime, EstadoRisco]] = _ListaDescartavel() if _somente_series else []
    conferencias: list[dict[str, Any]] = _ListaDescartavel() if _somente_series else []
    blocos: list[str] = []
    pnl_dia: defaultdict[date, float] = defaultdict(float)
    expo_compra: defaultdict[date, float] = defaultdict(float)
    expo_venda: defaultdict[date, float] = defaultdict(float)
    realizado_dia: defaultdict[date, float] = defaultdict(float)
    capital_inicio_dia: dict[date, float] = {}
    bloqueado_dia: set[date] = set()
    risco_dia_cache: dict[date, float] = {}
    reducao_no_instante: set[datetime] = set()
    equity_acumulada = params.capital_usd
    pico = params.capital_usd
    estado = EstadoRisco.NORMAL
    revogacao_alarmada = False
    dia_estado: date | None = None
    instante_atual: datetime | None = None
    versao_estado = 0
    cache_snapshot: tuple[tuple[int, datetime | None, date | None], tuple[float, float, float]] | None = None

    def alarmar_revogacao(instante: datetime, origem: str) -> None:
        """Publica a primeira revogacao pela porta injetada, nunca pela rede."""
        nonlocal revogacao_alarmada
        if alarme is None or revogacao_alarmada:
            return
        evento = {
            "tipo": "revogacao",
            "instante": instante,
            "estado": EstadoRisco.REVOGADO.value,
            "origem": origem,
            "capital_constante": params.capital_constante,
        }
        enviar = getattr(alarme, "enviar", None)
        if enviar is not None:
            enviar(evento)
        elif callable(alarme):
            alarme(evento)
        else:
            raise TypeError("alarme deve ser callable ou expor enviar(evento)")
        revogacao_alarmada = True

    def invalidar_snapshot() -> None:
        nonlocal versao_estado, cache_snapshot
        versao_estado += 1
        cache_snapshot = None

    def exposicao_conhecida(instante: datetime) -> bool:
        fn = getattr(carregador, "exposicao_conhecida", None)
        return True if fn is None else bool(fn(instante))

    def snapshot(instante: datetime | None = None) -> tuple[float, float, float]:
        """Margem, patrimônio marcado e risco; memoizado entre mutações."""
        nonlocal cache_snapshot
        instante = instante if instante is not None else instante_atual
        chave = (versao_estado, instante, dia_estado)
        if _otimizar and cache_snapshot is not None and cache_snapshot[0] == chave:
            return cache_snapshot[1]
        if not posicoes:
            usada = risco = aberto = 0
        elif len(posicoes) == 1:
            p = posicoes[0]
            usada = _margem_posicao_por_lote(p, instante) * p.quantidade
            risco = _risco_atual(
                p, p.ultimo_bid, p.ultimo_spread, p.ultimo_cambio_usd
            )
            aberto = _pnl(
                p,
                _preco_saida(
                    p, p.ultimo_bid, p.ultimo_spread,
                    spread_real=p.ultimo_spread_real,
                ),
                p.ultimo_cambio_usd,
            )
        else:
            usada = risco = aberto = 0.0
            for p in posicoes:
                usada += _margem_posicao_por_lote(p, instante) * p.quantidade
                risco += _risco_atual(
                    p, p.ultimo_bid, p.ultimo_spread, p.ultimo_cambio_usd,
                )
                aberto += _pnl(
                    p,
                    _preco_saida(
                        p, p.ultimo_bid, p.ultimo_spread,
                        spread_real=p.ultimo_spread_real,
                    ),
                    p.ultimo_cambio_usd,
                )
        if params.capital_constante:
            realizado = realizado_dia[dia_estado] if dia_estado is not None else 0.0
            equity = params.capital_usd + realizado + aberto
        else:
            equity = equity_acumulada + aberto
        valor = (usada, equity, risco)
        if _otimizar:
            cache_snapshot = (chave, valor)
        return valor

    def margem_usada(instante: datetime) -> float:
        return snapshot(instante)[0]

    def risco_aberto() -> float:
        return snapshot()[2]

    def marcar_equity() -> float:
        return snapshot()[1]

    def conferir_margem(instante: datetime, origem: str) -> tuple[float, float, float]:
        usada, e, risco = snapshot(instante)
        e_stop = e - risco
        fracao = usada / e_stop if e_stop > 0 else float("inf")
        conferencias.append({
            "instante": instante, "origem": origem, "margem_usd": usada,
            "patrimonio_estressado_usd": e_stop, "fracao": fracao,
            "limite": params.margem_max,
        })
        return usada, e_stop, fracao

    def fechar(pos: _Posicao, instante: datetime, bid: int, spread: int, motivo: str, dia: date,
               *, preco_forcado: int | None = None, derrapagem: bool = False) -> None:
        nonlocal equity_acumulada
        saida = preco_forcado if preco_forcado is not None else _preco_saida(
            pos, bid, spread, slippage=derrapagem, spread_real=pos.ultimo_spread_real
        )
        ganho = _pnl(pos, saida, pos.ultimo_cambio_usd)
        equity_acumulada += ganho
        pnl_dia[dia] += ganho
        realizado_dia[dia] += ganho
        if not _somente_series:
            livro.append({
                "evento": "fechamento", "intencao_id": pos.intencao.intencao_id,
                "instante": instante, "preco_saida": saida, "motivo": motivo,
                "pnl_usd": ganho, "quantidade": pos.quantidade,
            })
        posicoes.remove(pos)
        risco_dia_cache.pop(date.fromisoformat(pos.bloco), None)
        invalidar_snapshot()

    def risco_contratado_no_dia(dia: date) -> float:
        valor = risco_dia_cache.get(dia)
        if valor is None:
            valor = sum(
                p.risco_entrada_usd
                for p in posicoes if date.fromisoformat(p.bloco) == dia
            )
            risco_dia_cache[dia] = valor
        return valor

    def liquidar_por_margem(instante: datetime, barras_agora: Mapping[str, BarraMotor]) -> None:
        # Stop-out: maior perda aberta primeiro; reducao de margem: maior contribuicao primeiro.
        while posicoes:
            usada, e_stop, fracao = conferir_margem(instante, "controle_margem")
            nivel_margem = e_stop / usada if usada > 0 else float("inf")
            stopout = max(
                (p.especificacao.stop_out_nivel_margem for p in posicoes), default=0.0
            )
            if fracao < params.margem_max:
                return
            motivo = "stop_out" if nivel_margem <= stopout else "reducao_margem"
            if motivo == "stop_out":
                pos = min(
                    posicoes,
                    key=lambda p: _pnl(
                        p,
                        _preco_saida(
                            p, p.ultimo_bid, p.ultimo_spread,
                            spread_real=p.ultimo_spread_real,
                        ),
                        p.ultimo_cambio_usd,
                    ),
                )
            else:
                pos = max(
                    posicoes,
                    key=lambda p: _margem_posicao_por_lote(p, instante) * p.quantidade,
                )
            b = barras_agora.get(pos.intencao.simbolo)
            dia = b.data_servidor if b else instante.date()
            fechar(pos, instante, pos.ultimo_bid, pos.ultimo_spread, motivo, dia, derrapagem=True)

    def atualizar_estado(instante: datetime, barras_agora: Mapping[str, BarraMotor]) -> None:
        nonlocal estado, pico
        estado_anterior = estado
        origem_revogacao = "controle_risco"
        if not exposicao_conhecida(instante):
            estado = EstadoRisco.DESCONHECIDA
            estados.append((instante, estado))
            return
        if estado is EstadoRisco.DESCONHECIDA:
            estado = EstadoRisco.NORMAL
        e = marcar_equity()
        if not params.capital_constante:
            pico = max(pico, e)
        piso = (1.0 - params.limiar_m) * pico
        l = e - risco_aberto()
        if estado is EstadoRisco.REVOGADO:
            return
        if e <= piso:
            estado = EstadoRisco.REVOGADO
            origem_revogacao = "patrimonio_abaixo_piso"
        elif l < piso:
            estado = EstadoRisco.REDUCAO
            reducao_no_instante.add(instante)
            # Ordem fixa: maior contribuicao para o risco primeiro.
            while posicoes and marcar_equity() - risco_aberto() < piso:
                pos = max(
                    posicoes,
                    key=lambda p: _risco_atual(
                        p, p.ultimo_bid, p.ultimo_spread, p.ultimo_cambio_usd
                    ),
                )
                b = barras_agora.get(pos.intencao.simbolo)
                fechar(pos, instante, pos.ultimo_bid, pos.ultimo_spread, "reducao_risco", b.data_servidor if b else instante.date())
            e = marcar_equity()
            if e <= piso:
                estado = EstadoRisco.REVOGADO
                origem_revogacao = "reducao_insuficiente"
            elif e - risco_aberto() >= piso:
                estado = EstadoRisco.NORMAL
        else:
            estado = EstadoRisco.NORMAL
        if estado is not EstadoRisco.REVOGADO and not posicoes:
            menor = 0.0 if lote_continuo_diagnostico else min(
                (
                    s.stop_minimo_pontos * s.valor_ponto_moeda_cotacao * s.lote_minimo
                    * (barras_agora[nome].cambio_usd if nome in barras_agora else 1.0)
                    for nome, s in specs.items()
                ),
                default=0.0,
            )
            if e - piso < menor - 1e-12:
                estado = EstadoRisco.REVOGADO
                origem_revogacao = "fim_folga"
        if estado is EstadoRisco.REVOGADO:
            for pos in list(posicoes):
                b = barras_agora.get(pos.intencao.simbolo)
                fechar(pos, instante, pos.ultimo_bid, pos.ultimo_spread, "revogado", b.data_servidor if b else instante.date(), derrapagem=True)
        if estado is EstadoRisco.REVOGADO and estado_anterior is not EstadoRisco.REVOGADO:
            alarmar_revogacao(instante, origem_revogacao)
        estados.append((instante, estado))

    def barra_vazia_constante(
        instante: datetime, barras_agora: Mapping[str, BarraMotor]
    ) -> None:
        """Caminho O(1) sem posição/intenção, com saída idêntica ao laço geral."""
        nonlocal estado
        e = params.capital_usd + realizado_dia[dia_estado]

        def margem(origem: str) -> None:
            conferencias.append({
                "instante": instante,
                "origem": origem,
                "margem_usd": 0,
                "patrimonio_estressado_usd": e,
                "fracao": 0.0 if e > 0 else float("inf"),
                "limite": params.margem_max,
            })

        def estado_vazio() -> None:
            nonlocal estado
            estado_anterior = estado
            if not exposicao_conhecida(instante):
                estado = EstadoRisco.DESCONHECIDA
                estados.append((instante, estado))
                return
            if estado is EstadoRisco.DESCONHECIDA:
                estado = EstadoRisco.NORMAL
            piso = (1.0 - params.limiar_m) * pico
            if estado is EstadoRisco.REVOGADO:
                return
            estado = EstadoRisco.REVOGADO if e <= piso else EstadoRisco.NORMAL
            if estado is not EstadoRisco.REVOGADO:
                menor = 0.0 if lote_continuo_diagnostico else min(
                    (
                        s.stop_minimo_pontos * s.valor_ponto_moeda_cotacao
                        * s.lote_minimo
                        * (barras_agora[nome].cambio_usd if nome in barras_agora else 1.0)
                        for nome, s in specs.items()
                    ),
                    default=0.0,
                )
                if e - piso < menor - 1e-12:
                    estado = EstadoRisco.REVOGADO
            if estado is EstadoRisco.REVOGADO and estado_anterior is not EstadoRisco.REVOGADO:
                alarmar_revogacao(instante, "fim_folga")
            estados.append((instante, estado))

        margem("barra_antes")
        estado_vazio()
        for dia in {b.data_servidor for b in barras_agora.values()}:
            capital_inicio_dia.setdefault(dia, params.capital_usd)
            perda = max(0.0, -realizado_dia[dia])
            if perda > params.limite_perda_dia * capital_inicio_dia[dia] + 1e-12:
                bloqueado_dia.add(dia)
        margem("barra_depois")
        estado_vazio()

    def controles_normais_constantes(
        instante: datetime, barras_agora: Mapping[str, BarraMotor], origem: str
    ) -> bool:
        """Materializa controles sem repetir cálculos quando nada pode mudar de estado."""
        nonlocal estado
        if estado is EstadoRisco.REVOGADO or not exposicao_conhecida(instante):
            return False
        usada, e, risco = snapshot(instante)
        e_stop = e - risco
        fracao = usada / e_stop if e_stop > 0 else float("inf")
        piso = (1.0 - params.limiar_m) * pico
        if fracao >= params.margem_max or e <= piso or e_stop < piso:
            return False
        if not posicoes:
            menor = 0.0 if lote_continuo_diagnostico else min(
                (
                    s.stop_minimo_pontos * s.valor_ponto_moeda_cotacao
                    * s.lote_minimo
                    * (barras_agora[nome].cambio_usd if nome in barras_agora else 1.0)
                    for nome, s in specs.items()
                ),
                default=0.0,
            )
            if e - piso < menor - 1e-12:
                return False
        estado = EstadoRisco.NORMAL
        registro = {
            "instante": instante,
            "origem": origem,
            "margem_usd": usada,
            "patrimonio_estressado_usd": e_stop,
            "fracao": fracao,
            "limite": params.margem_max,
        }
        conferencias.append(registro)
        if posicoes:
            conferencias.append({**registro, "origem": "controle_margem"})
        estados.append((instante, estado))
        return True

    def controles_posicao_simples(
        instante: datetime, origem: str, p: _Posicao
    ) -> bool:
        """Controle equivalente para uma posicao sem fonte externa de exposicao."""
        nonlocal estado
        if (
            estado is EstadoRisco.REVOGADO
            or getattr(carregador, "exposicao_conhecida", None) is not None
        ):
            return False
        usada = _margem_posicao_por_lote(p, instante) * p.quantidade
        risco = _risco_atual(p, p.ultimo_bid, p.ultimo_spread, p.ultimo_cambio_usd)
        aberto = _pnl(
            p,
            _preco_saida(
                p, p.ultimo_bid, p.ultimo_spread,
                spread_real=p.ultimo_spread_real,
            ),
            p.ultimo_cambio_usd,
        )
        e = params.capital_usd + realizado_dia[dia_estado] + aberto
        e_stop = e - risco
        fracao = usada / e_stop if e_stop > 0 else float("inf")
        piso = (1.0 - params.limiar_m) * pico
        if fracao >= params.margem_max or e <= piso or e_stop < piso:
            return False
        estado = EstadoRisco.NORMAL
        registro = {
            "instante": instante,
            "origem": origem,
            "margem_usd": usada,
            "patrimonio_estressado_usd": e_stop,
            "fracao": fracao,
            "limite": params.margem_max,
        }
        conferencias.append(registro)
        conferencias.append({**registro, "origem": "controle_margem"})
        estados.append((instante, estado))
        return True

    for instante in sorted(set(barras_evento) | set(eventos) | set(faltas_cofre)):
        if (
            _otimizar and _somente_series and not posicoes and not eventos.get(instante)
            and not faltas_cofre.get(instante)
        ):
            continue
        instante_atual = instante
        barras_agora = {b.simbolo: b for b in barras_evento.get(instante, ())}
        dias_instante = {b.data_servidor for b in barras_agora.values()}
        if not dias_instante:
            dias_instante = {b.data_servidor for _, b, _ in eventos.get(instante, ())}
        if not dias_instante:
            dias_instante = {d for _, d, _ in faltas_cofre.get(instante, ())}
        novo_dia = min(dias_instante) if dias_instante else instante.date()
        if params.capital_constante and novo_dia != dia_estado:
            # F2-1b: cada dia da serie de prova e uma nova conta de US$ 3.000.
            # Revogacao, pico e lucro/prejuizo de ontem nao atravessam a fronteira.
            dia_estado = novo_dia
            pico = params.capital_usd
            estado = EstadoRisco.NORMAL
        elif dia_estado is None:
            dia_estado = novo_dia
        # Atualiza a marca para a abertura antes de qualquer controle.
        for p in posicoes:
            b = barras_agora.get(p.intencao.simbolo)
            if b is not None:
                p.ultimo_bid = b.abertura
                p.ultimo_spread_real = b.spread
                p.ultimo_spread = spread_barra(b)
                p.ultimo_cambio_usd = b.cambio_usd
        invalidar_snapshot()

        # D0(b): a sessão prova que deveria existir pregão, mas o símbolo não
        # tem uma única M15 no dia do Cofre. Não se carrega a posição até o
        # fim do arquivo: realiza-se o stop no primeiro dia faltante. A série
        # pessimista acrescenta o pior spread causal da mesma hora.
        for simbolo_falta, dia_falta, _ in faltas_cofre.get(instante, ()):
            for p in [
                x for x in posicoes if x.intencao.simbolo == simbolo_falta
            ]:
                spread_execucao = p.ultimo_spread
                if params.pessimista:
                    pior = spreads_falta[(simbolo_falta, dia_falta)]
                    if pior is not None:
                        spread_execucao = max(spread_execucao, pior)
                delta = p.especificacao.slippage_pontos + max(
                    0, spread_execucao - p.ultimo_spread_real
                )
                preco = (
                    p.stop - delta
                    if p.intencao.lado is Lado.COMPRA
                    else p.stop + delta
                )
                fechar(
                    p, instante, p.ultimo_bid, spread_execucao,
                    "cofre_sem_m15_stop", dia_falta, preco_forcado=preco,
                )

        if (
            _otimizar
            and params.capital_constante
            and not posicoes
            and not eventos.get(instante)
        ):
            # Caso dominante entre campanhas: sem posicao, sem exposicao externa
            # desconhecida e longe de qualquer fronteira de risco. Materializa os
            # mesmos registros sem chamadas e agregacoes redundantes.
            if getattr(carregador, "exposicao_conhecida", None) is None:
                e_vazio = params.capital_usd + realizado_dia[dia_estado]
                piso_vazio = (1.0 - params.limiar_m) * pico
                menor_vazio = 0.0 if lote_continuo_diagnostico else min(
                    (
                        s.stop_minimo_pontos * s.valor_ponto_moeda_cotacao
                        * s.lote_minimo
                        * (barras_agora[nome].cambio_usd if nome in barras_agora else 1.0)
                        for nome, s in specs.items()
                    ),
                    default=0.0,
                )
                if (
                    estado is not EstadoRisco.REVOGADO
                    and e_vazio > piso_vazio
                    and e_vazio - piso_vazio >= menor_vazio - 1e-12
                ):
                    registro_vazio = {
                        "instante": instante,
                        "origem": "barra_antes",
                        "margem_usd": 0,
                        "patrimonio_estressado_usd": e_vazio,
                        "fracao": 0.0 if e_vazio > 0 else float("inf"),
                        "limite": params.margem_max,
                    }
                    conferencias.append(registro_vazio)
                    estados.append((instante, EstadoRisco.NORMAL))
                    for dia in dias_instante:
                        capital_inicio_dia.setdefault(dia, params.capital_usd)
                        perda = max(0.0, -realizado_dia[dia])
                        if perda > params.limite_perda_dia * capital_inicio_dia[dia] + 1e-12:
                            bloqueado_dia.add(dia)
                    conferencias.append({**registro_vazio, "origem": "barra_depois"})
                    estados.append((instante, EstadoRisco.NORMAL))
                    estado = EstadoRisco.NORMAL
                    continue
            barra_vazia_constante(instante, barras_agora)
            continue

        if (
            _otimizar
            and params.capital_constante
            and len(posicoes) == 1
            and not eventos.get(instante)
            and not controles_implicitos
        ):
            p_rapida = posicoes[0]
            b_rapida = barras_agora.get(p_rapida.intencao.simbolo)
            if (
                b_rapida is not None
                and instante < p_rapida.intencao.hora_limite
                and not p_rapida.especificacao.ajustes
                and (
                    controles_posicao_simples(instante, "barra_antes", p_rapida)
                    or controles_normais_constantes(
                        instante, barras_agora, "barra_antes"
                    )
                )
            ):
                spread_saida = spread_barra(b_rapida)
                barra_custo = (
                    b_rapida if spread_saida == b_rapida.spread
                    else replace(b_rapida, spread=spread_saida)
                )
                gatilho = _atingimento(
                    p_rapida, barra_custo, spread_real=b_rapida.spread
                )
                if gatilho is not None:
                    motivo, preco = gatilho
                    p_rapida.ultimo_cambio_usd = b_rapida.cambio_usd
                    fechar(
                        p_rapida, instante, b_rapida.fechamento, spread_saida,
                        motivo, b_rapida.data_servidor, preco_forcado=preco,
                    )
                else:
                    p_rapida.ultimo_bid = b_rapida.fechamento
                    p_rapida.ultimo_spread = spread_saida
                    p_rapida.ultimo_spread_real = b_rapida.spread
                    p_rapida.ultimo_cambio_usd = b_rapida.cambio_usd
                    invalidar_snapshot()
                dia_rapido = b_rapida.data_servidor
                capital_inicio_dia.setdefault(dia_rapido, params.capital_usd)
                aberto_dia = risco_contratado_no_dia(dia_rapido)
                perda = max(0.0, -realizado_dia[dia_rapido]) + aberto_dia
                if perda > params.limite_perda_dia * capital_inicio_dia[dia_rapido] + 1e-12:
                    bloqueado_dia.add(dia_rapido)
                    for p in list(posicoes):
                        if date.fromisoformat(p.bloco) == dia_rapido:
                            fechar(
                                p, instante, p.ultimo_bid, p.ultimo_spread,
                                "limite_dia", dia_rapido, derrapagem=True,
                            )
                controle_depois = (
                    bool(posicoes)
                    and controles_posicao_simples(
                        instante, "barra_depois", posicoes[0]
                    )
                ) or controles_normais_constantes(
                    instante, barras_agora, "barra_depois"
                )
                if not controle_depois:
                    conferir_margem(instante, "barra_depois")
                    liquidar_por_margem(instante, barras_agora)
                    atualizar_estado(instante, barras_agora)
                continue

        # Ajustes de contrato no instante. Day trade nunca paga swap.
        for p in list(posicoes):
            for a in p.especificacao.ajustes:
                dividendo_valido = a.tipo != "dividendo" or p.especificacao.modo_contrato in {
                    "spot", "indice_spot", "acao"
                }
                swap_valido = a.tipo != "swap" or (
                    p.intencao.campanha is not Campanha.DAY_TRADE_M15
                    and p.especificacao.modo_contrato != "futuro"
                )
                if a.instante == instante and dividendo_valido and swap_valido:
                    sinal = 1.0 if p.intencao.lado is Lado.COMPRA else -1.0
                    valor = sinal * a.valor_por_lote * a.cambio_usd * p.quantidade
                    p.ajustes_usd += valor
                    invalidar_snapshot()
                    livro.append({"evento": a.tipo, "intencao_id": p.intencao.intencao_id,
                                  "instante": instante, "valor_usd": valor})

        # C7: fecha na abertura da fronteira; se ela faltou, usa a proxima barra com custo pessimista.
        for p in list(posicoes):
            b = barras_agora.get(p.intencao.simbolo)
            if b is not None and instante >= p.intencao.hora_limite:
                spread_saida = spread_barra(b)
                fechar(p, instante, b.abertura, spread_saida, "zeragem", b.data_servidor,
                       derrapagem=True)

        if controles_implicitos:
            pass
        elif not (
            (_somente_series and not posicoes)
            or (
                _otimizar
                and params.capital_constante
                and controles_normais_constantes(
                    instante, barras_agora, "barra_antes"
                )
            )
        ):
            conferir_margem(instante, "barra_antes")
            liquidar_por_margem(instante, barras_agora)
            atualizar_estado(instante, barras_agora)
        # Admissoes na abertura da barra. No caminho de séries, uma posição
        # aberta torna todas as demais intenções do símbolo inelegíveis; o set
        # evita materializar recusas e repetir toda a cadeia econômica.
        simbolos_abertos = (
            {p.intencao.simbolo for p in posicoes} if _somente_series else set()
        )
        itens_admissao = eventos.get(instante, ())
        ordem_admissao = (
            _admissoes_enxutas(itens_admissao, simbolos_abertos)
            if _somente_series else
            sorted(
                itens_admissao,
                key=lambda x: (-x[0].peso_admissao, x[0].intencao_id),
            )
        )
        for i, b, fallback_diario in ordem_admissao:
            dia = b.data_servidor
            if _somente_series and (
                estado in {
                    EstadoRisco.REVOGADO,
                    EstadoRisco.DESCONHECIDA,
                    EstadoRisco.REDUCAO,
                }
                or instante in reducao_no_instante
                or dia in bloqueado_dia
                or i.simbolo in simbolos_abertos
            ):
                continue
            spec = specs[i.simbolo]
            capital_inicio_dia.setdefault(
                dia, params.capital_usd if params.capital_constante else marcar_equity()
            )
            try:
                s = _sessao(calendario, i.simbolo, dia)
            except (KeyError, TypeError):
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.SEM_SESSAO, {"dia": dia.isoformat()}))
                continue
            limite = min(i.hora_limite, s.limite_zeragem_utc or i.hora_limite, s.virada_utc)
            if instante >= limite:
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.ZERAGEM, {
                    "entrada": instante.isoformat(), "limite_efetivo": limite.isoformat(),
                    "relacao": "entrada >= limite_efetivo",
                }))
                continue
            if estado is EstadoRisco.REVOGADO:
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.REVOGADO, {"estado": estado.value}))
                continue
            if estado is EstadoRisco.DESCONHECIDA:
                recusas.append(Recusa(
                    i.intencao_id, instante, MotivoRecusa.EXPOSICAO_DESCONHECIDA,
                    {"estado": estado.value, "relacao": "exposicao nao reconciliada"},
                ))
                continue
            if estado is EstadoRisco.REDUCAO or instante in reducao_no_instante:
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.REDUCAO, {
                    "estado": EstadoRisco.REDUCAO.value,
                    "relacao": "L < F exigiu reducao neste instante",
                }))
                continue
            if dia in bloqueado_dia:
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.LIMITE_DIA, {"dia_bloqueado": True}))
                continue
            aberta = next(
                (p for p in posicoes if p.intencao.simbolo == i.simbolo), None
            )
            if aberta is not None:
                recusas.append(Recusa(
                    i.intencao_id,
                    instante,
                    MotivoRecusa.POSICAO_ABERTA,
                    {
                        "simbolo": i.simbolo,
                        "intencao_aberta": aberta.intencao.intencao_id,
                        "lado_aberto": aberta.intencao.lado.value,
                        "lado_novo": i.lado.value,
                        "margin_mode": 0,
                        "relacao": "conta NETTING admite uma posicao por simbolo",
                    },
                ))
                continue

            indicador = indicador_barra(b)
            if indicador.spread_mediano is None or indicador.movimento_mediano is None:
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.FILTRO_HORA, {
                    "observacoes": indicador.observacoes,
                    "exigidas": params.filtro_janela_pregoes,
                    "relacao": "observacoes < exigidas",
                }))
                continue
            spread_med, mov_med = indicador.spread_mediano, indicador.movimento_mediano
            if spread_med > params.filtro_spread_mov * mov_med:
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.FILTRO_HORA, {
                    "spread_mediano": spread_med, "movimento_mediano": mov_med,
                    "limite": params.filtro_spread_mov * mov_med,
                    "relacao": "spread_mediano > limite",
                }))
                continue

            spread = spread_barra(b)
            risco_lote = (
                (i.stop_pontos + spec.slippage_pontos)
                * spec.valor_ponto_moeda_cotacao * b.cambio_usd
            )
            base_tamanho = (
                params.capital_usd if params.capital_constante else capital_inicio_dia[dia]
            )
            maximo = params.risco_por_operacao * base_tamanho
            quantidade = _quantidade(
                maximo / risco_lote, spec, lote_continuo_diagnostico
            )
            if (
                not lote_continuo_diagnostico
                and quantidade < spec.lote_minimo
            ):
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.LOTE_MINIMO, {
                    "quantidade_calculada": quantidade, "lote_minimo": spec.lote_minimo,
                }))
                continue
            risco_novo = risco_lote * quantidade
            e = marcar_equity()
            if not params.capital_constante:
                pico = max(pico, e)
            pico_referencia = params.capital_usd if params.capital_constante else pico
            piso = (1.0 - params.limiar_m) * pico_referencia
            aberto = risco_aberto()
            alvo_risco = params.risco_aberto_alvo * params.limiar_m * pico_referencia
            if aberto + risco_novo > alvo_risco + 1e-9:
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.RISCO_ABERTO, {
                    "risco_aberto": aberto, "risco_novo": risco_novo, "alvo": alvo_risco,
                    "relacao": "risco_aberto + risco_novo > alvo",
                }))
                continue
            if e - aberto - risco_novo < piso - 1e-9:
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.PISO_F, {
                    "L_depois": e - aberto - risco_novo, "F": piso, "relacao": "L_depois < F",
                }))
                continue
            risco_classe = sum(
                _risco_atual(
                    p, p.ultimo_bid, p.ultimo_spread, p.ultimo_cambio_usd
                )
                for p in posicoes if p.especificacao.limita_um_terco
            )
            limite_classe = params.fracao_um_terco * params.limiar_m * pico_referencia
            if spec.limita_um_terco and risco_classe + risco_novo > limite_classe + 1e-9:
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.UM_TERCO, {
                    "risco_classe": risco_classe, "risco_novo": risco_novo, "limite": limite_classe,
                    "relacao": "risco_classe + risco_novo > limite",
                }))
                continue
            perda_realizada = max(0.0, -realizado_dia[dia])
            perda_aberta_dia = risco_contratado_no_dia(dia)
            limite_dia = params.limite_perda_dia * capital_inicio_dia[dia]
            if perda_realizada + perda_aberta_dia + risco_novo >= limite_dia - 1e-12:
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.LIMITE_DIA, {
                    "perda_realizada": perda_realizada, "perda_aberta_stops": perda_aberta_dia,
                    "risco_nova": risco_novo, "limite": limite_dia,
                    "relacao": "soma >= limite (igualdade rejeita)",
                }))
                continue
            entrada = _preco_entrada(i, b, spread, spec)
            margem_nova = spec.margem_no_instante(
                instante, entrada, b.cambio_usd
            ) * quantidade
            usada_depois = margem_usada(instante) + margem_nova
            estressado_depois = e - aberto - risco_novo
            fracao_depois = usada_depois / estressado_depois if estressado_depois > 0 else float("inf")
            if not _somente_series:
                conferencias.append({
                    "instante": instante, "origem": "ordem_antes", "intencao_id": i.intencao_id,
                    "margem_usd": usada_depois, "patrimonio_estressado_usd": estressado_depois,
                    "fracao": fracao_depois, "limite": params.margem_max,
                })
            if fracao_depois > params.margem_max + 1e-12:
                recusas.append(Recusa(i.intencao_id, instante, MotivoRecusa.MARGEM, {
                    "margem_depois": usada_depois, "patrimonio_estressado": estressado_depois,
                    "fracao": fracao_depois, "limite": params.margem_max,
                    "relacao": "fracao > limite",
                }))
                continue
            stop = entrada - i.stop_pontos if i.lado is Lado.COMPRA else entrada + i.stop_pontos
            alvo = None if i.alvo_pontos is None else (
                entrada + i.alvo_pontos if i.lado is Lado.COMPRA else entrada - i.alvo_pontos
            )
            intencao_efetiva = _com_hora_limite(i, limite)
            pos = _Posicao(intencao_efetiva, spec, quantidade, instante, entrada, stop, alvo, spread,
                           risco_novo, margem_nova, dia.isoformat(), fallback_diario,
                           ultimo_bid=b.abertura, ultimo_spread=spread,
                           ultimo_spread_real=b.spread, ultimo_cambio_usd=b.cambio_usd)
            posicoes.append(pos)
            risco_dia_cache.pop(dia, None)
            if _somente_series:
                simbolos_abertos.add(i.simbolo)
            invalidar_snapshot()
            if fallback_diario and spec.custo_convencao_diaria_usd_por_lote:
                pos.ajustes_usd -= spec.custo_convencao_diaria_usd_por_lote * quantidade
            if (params.pessimista and spec.limita_um_terco
                    and spec.dividendo_pessimista_usd_por_lote_dia
                    and not any(a.tipo == "dividendo" and a.instante.date() == dia for a in spec.ajustes)):
                pos.ajustes_usd -= spec.dividendo_pessimista_usd_por_lote_dia * quantidade
            blocos.append(pos.bloco)
            notional = abs(
                entrada * spec.valor_ponto_moeda_cotacao * quantidade * b.cambio_usd
            )
            (expo_compra if i.lado is Lado.COMPRA else expo_venda)[dia] += notional
            if not _somente_series:
                livro.append({
                    "evento": "abertura", "intencao_id": i.intencao_id, "instante": instante,
                    "simbolo": i.simbolo, "lado": i.lado.value, "preco_entrada": entrada,
                    "spread": spread, "stop": stop, "alvo": alvo, "quantidade": quantidade,
                    "risco_usd": risco_novo, "margem_usd": margem_nova,
                    "fallback_diario": fallback_diario,
                })
            if not _somente_series:
                conferir_margem(instante, "ordem_depois")

        # Caminho dentro da barra, conservador se nao houver detalhe.
        for p in list(posicoes):
            b = barras_agora.get(p.intencao.simbolo)
            if b is None:
                continue
            spread_saida = spread_barra(b)
            barra_custo = b if spread_saida == b.spread else replace(b, spread=spread_saida)
            gatilho = _atingimento(p, barra_custo, spread_real=b.spread)
            if gatilho is not None:
                motivo, preco = gatilho
                p.ultimo_cambio_usd = b.cambio_usd
                fechar(
                    p, instante, b.fechamento, spread_saida, motivo, b.data_servidor,
                    preco_forcado=preco,
                )
            else:
                p.ultimo_bid = b.fechamento
                p.ultimo_spread = spread_saida
                p.ultimo_spread_real = b.spread
                p.ultimo_cambio_usd = b.cambio_usd
                invalidar_snapshot()

        # R6: o limite diario usa a perda contratada desde a entrada ate os
        # stops, nao a perda marcada. Gaps realizados entram imediatamente.
        dias_agora = {b.data_servidor for b in barras_agora.values()}
        for dia in dias_agora:
            capital_inicio_dia.setdefault(
                dia, params.capital_usd if params.capital_constante else marcar_equity()
            )
            aberto_dia = risco_contratado_no_dia(dia)
            perda = max(0.0, -realizado_dia[dia]) + aberto_dia
            if perda > params.limite_perda_dia * capital_inicio_dia[dia] + 1e-12:
                bloqueado_dia.add(dia)
                for p in list(posicoes):
                    b = barras_agora.get(p.intencao.simbolo)
                    if b is not None and b.data_servidor == dia:
                        fechar(p, instante, p.ultimo_bid, p.ultimo_spread, "limite_dia", dia, derrapagem=True)

        if controles_implicitos:
            pass
        elif not (
            (_somente_series and not posicoes)
            or (
                _otimizar
                and params.capital_constante
                and controles_normais_constantes(
                    instante, barras_agora, "barra_depois"
                )
            )
        ):
            conferir_margem(instante, "barra_depois")
            liquidar_por_margem(instante, barras_agora)
            atualizar_estado(instante, barras_agora)

    # Posicoes restantes sao encerradas no ultimo preco conhecido; day trade sem barra de limite
    # ja teria sido encerrado na primeira barra posterior. Ausencia total posterior e inconformidade.
    for p in list(posicoes):
        dia = (
            max(_precalculo.datas[p.intencao.simbolo], default=p.entrada_em.date())
            if _precalculo is not None else
            max(
                (b.data_servidor for b in barras_por_simbolo[p.intencao.simbolo]),
                default=p.entrada_em.date(),
            )
        )
        delta = p.especificacao.slippage_pontos + max(
            0, p.ultimo_spread - p.ultimo_spread_real
        )
        stop_pessimista = (
            p.stop - delta if p.intencao.lado is Lado.COMPRA else p.stop + delta
        )
        fechar(p, p.intencao.hora_limite, p.ultimo_bid, p.ultimo_spread,
               "zeragem_sem_barra_posterior", dia, preco_forcado=stop_pessimista)

    lotes_inicio = _lotes_compactos(intencoes)
    if lotes_inicio is not None:
        primeiro_ns = min(
            (int(l.sinal_fim_ns[0]) for l in lotes_inicio if len(l)),
            default=None,
        )
        inicio_prova = (
            _datetime_utc(primeiro_ns).date() if primeiro_ns is not None
            else date.max
        )
    else:
        inicio_prova = min(
            (i.barra_sinal_fim.date() for i in intencoes), default=date.max,
        )
    datas_base = (
        set().union(*(_precalculo.datas[s] for s in simbolos))
        if _precalculo is not None and simbolos else
        {b.data_servidor for barras in barras_por_simbolo.values() for b in barras}
    )
    datas = sorted(d for d in (datas_base | set(pnl_dia)) if d >= inicio_prova)
    pnl = tuple(float(pnl_dia[d]) for d in datas)
    return ResultadoMotor(
        tuple(datas), pnl, tuple(x / params.capital_usd for x in pnl),
        tuple(float(expo_compra[d]) for d in datas), tuple(float(expo_venda[d]) for d in datas),
        tuple(blocos), tuple(livro), tuple(recusas), tuple(estados), tuple(conferencias),
    )


def _preparar_par(
    intencoes: Sequence[Intencao], carregador: Any, calendario: Any,
    janela: int,
) -> tuple[
    tuple[str, ...], _PrecalculoCompartilhado,
    Mapping[datetime, Sequence[tuple[Intencao, BarraMotor, bool]]],
]:
    """Materializa uma vez tudo que não depende da máquina econômica."""
    simbolos = tuple(sorted({i.simbolo for i in intencoes}))
    barras_por_simbolo = {s: _barras_do(carregador, s) for s in simbolos}
    indices = {s: _indexar_barras(bs) for s, bs in barras_por_simbolo.items()}
    barras_evento_mut: defaultdict[datetime, list[BarraMotor]] = defaultdict(list)
    for barras in barras_por_simbolo.values():
        dias_com_intradia = {
            b.data_servidor for b in barras if b.tempo_grafico in {"M15", "H1"}
        }
        for b in barras:
            if b.tempo_grafico != "D1" or b.data_servidor not in dias_com_intradia:
                barras_evento_mut[b.inicio].append(b)
    barras_evento = {t: tuple(bs) for t, bs in barras_evento_mut.items()}
    # ``None`` calcula todas as horas, inclusive o máximo causal usado apenas
    # pela série pessimista; a série normal lê as mesmas medianas.
    indicadores = _precalcular_horas(
        barras_por_simbolo, calendario, janela, None,
    )
    precalculo = _PrecalculoCompartilhado(
        barras_por_simbolo,
        indices,
        barras_evento,
        indicadores,
        {s: bs[0].inicio for s, bs in barras_por_simbolo.items() if bs},
        {s: frozenset(b.data_servidor for b in bs) for s, bs in barras_por_simbolo.items()},
    )
    eventos_mut: defaultdict[
        datetime, list[tuple[Intencao, BarraMotor, bool]]
    ] = defaultdict(list)
    for i in intencoes:
        barra, fallback = _selecionar_entrada(i, indices[i.simbolo])
        entrada_prevista = (
            _ceil_m15(i.barra_sinal_fim)
            if i.campanha is Campanha.DAY_TRADE_M15
            else i.barra_sinal_fim
        )
        if (
            entrada_prevista < i.hora_limite
            and barra is not None
            and barra.inicio < i.hora_limite
        ):
            eventos_mut[barra.inicio].append((i, barra, fallback))
    eventos = {
        t: _preparar_admissoes(itens) for t, itens in eventos_mut.items()
    }
    return simbolos, precalculo, eventos


def _lotes_compactos(
    intencoes: Sequence[Intencao],
) -> tuple[LoteIntencoesCompactas, ...] | None:
    if isinstance(intencoes, LoteIntencoesCompactas):
        return (intencoes,)
    if isinstance(intencoes, ColecaoIntencoesCompactas):
        return intencoes.lotes
    return None


def _preparar_par_compacto(
    intencoes: Sequence[Intencao], carregador: Any, calendario: Any,
    janela: int,
) -> tuple[
    tuple[str, ...], _PrecalculoCompartilhado,
    Mapping[datetime, _AdmissoesPreparadas],
]:
    """Prepara lotes M15 sem criar uma visao Python para cada sinal.

    Dentro de cada instante o lote ja esta em ``intencao_id`` crescente. Uma
    ordenacao estavel apenas por peso produz exatamente ``(-peso, id)``.
    """
    lotes = _lotes_compactos(intencoes)
    if lotes is None or any(l.campanha is not Campanha.DAY_TRADE_M15 for l in lotes):
        return _preparar_par(intencoes, carregador, calendario, janela)
    simbolos = tuple(sorted(l.simbolo for l in lotes))
    barras_por_simbolo = {s: _barras_do(carregador, s) for s in simbolos}
    indices = {s: _indexar_barras(bs) for s, bs in barras_por_simbolo.items()}
    barras_evento_mut: defaultdict[datetime, list[BarraMotor]] = defaultdict(list)
    for barras in barras_por_simbolo.values():
        dias_com_intradia = {
            b.data_servidor for b in barras if b.tempo_grafico in {"M15", "H1"}
        }
        for b in barras:
            if b.tempo_grafico != "D1" or b.data_servidor not in dias_com_intradia:
                barras_evento_mut[b.inicio].append(b)
    barras_evento = {t: tuple(bs) for t, bs in barras_evento_mut.items()}
    indicadores = _precalcular_horas(
        barras_por_simbolo, calendario, janela, None,
    )
    precalculo = _PrecalculoCompartilhado(
        barras_por_simbolo,
        indices,
        barras_evento,
        indicadores,
        {s: bs[0].inicio for s, bs in barras_por_simbolo.items() if bs},
        {s: frozenset(b.data_servidor for b in bs) for s, bs in barras_por_simbolo.items()},
    )
    por_instante: defaultdict[datetime, dict[str, _FilaCompacta]] = defaultdict(dict)
    for lote in lotes:
        n = len(lote)
        inicio = 0
        while inicio < n:
            sinal_ns = int(lote.sinal_fim_ns[inicio])
            fim = int(np.searchsorted(lote.sinal_fim_ns, sinal_ns, side="right"))
            instante = _ceil_m15(_datetime_utc(sinal_ns))
            barra = indices[lote.simbolo].m15_por_inicio.get(instante)
            if barra is not None:
                linhas = np.arange(inicio, fim, dtype=np.int64)
                limites = lote.hora_limite_ns[linhas]
                delta = instante - datetime(1970, 1, 1, tzinfo=UTC)
                instante_ns = (
                    (delta.days * 86_400 + delta.seconds) * 1_000_000_000
                    + delta.microseconds * 1000
                )
                linhas = linhas[limites > instante_ns]
                if len(linhas):
                    ordem = np.argsort(-lote.peso_admissao[linhas], kind="stable")
                    linhas = np.ascontiguousarray(linhas[ordem], dtype=np.int64)
                    anterior = por_instante[instante].get(lote.simbolo)
                    if anterior is not None:
                        # Sinais distintos podem cair na mesma abertura M15.
                        # Nesse caso a prioridade secundaria e o id global,
                        # nao a ordem dos instantes de sinal.
                        combinadas = np.concatenate((anterior.linhas, linhas))
                        combinadas = np.asarray(sorted(
                            map(int, combinadas),
                            key=lambda k: (
                                -float(lote.peso_admissao[k]),
                                lote.intencao_id(k),
                            ),
                        ), dtype=np.int64)
                        linhas = np.ascontiguousarray(combinadas)
                    por_instante[instante][lote.simbolo] = _FilaCompacta(
                        lote, linhas, barra,
                    )
            inicio = fim
    eventos = {
        instante: _AdmissoesPreparadas(dict(sorted(filas.items())))
        for instante, filas in por_instante.items()
    }
    return simbolos, precalculo, eventos


def simular_par(
    intencoes_por_barra: Mapping[datetime, Sequence[Intencao]] | Iterable[Intencao],
    carregador: Any,
    contratos: Any,
    calendario: Any,
    params_normal: ParamsMotor,
    params_pessimista: ParamsMotor,
    *,
    somente_series: bool = True,
) -> tuple[ResultadoMotor, ResultadoMotor]:
    """Roda normal e pessimista com uma preparação e estados independentes.

    Barras, índices, relógio de eventos, indicadores por hora, sessões de
    mercado implícitas no calendário e seleção das entradas são imutáveis e
    compartilhados. Cada chamada a :func:`simular` conserva posições, caixa,
    margem, estados e admissões próprios, pois custos pessimistas podem mudar
    toda a trajetória econômica.
    """
    if params_normal.pessimista or not params_pessimista.pessimista:
        raise ValueError("simular_par exige params normal e pessimista, nessa ordem")
    if replace(params_normal, pessimista=True) != params_pessimista:
        raise ValueError("os parâmetros do par só podem diferir em pessimista")
    if isinstance(intencoes_por_barra, Mapping):
        intencoes = [
            i for t in sorted(intencoes_por_barra) for i in intencoes_por_barra[t]
        ]
    elif eh_sequencia_compacta(intencoes_por_barra):
        intencoes = intencoes_por_barra
    else:
        recebidas = (
            intencoes_por_barra
            if isinstance(intencoes_por_barra, list)
            else list(intencoes_por_barra)
        )
        if recebidas and all(isinstance(x, LoteIntencoesCompactas) for x in recebidas):
            intencoes = ColecaoIntencoesCompactas(recebidas)
        elif any(isinstance(x, LoteIntencoesCompactas) for x in recebidas):
            raise TypeError("não misture lotes compactos e Intencao")
        else:
            intencoes = recebidas
    if (
        not eh_sequencia_compacta(intencoes)
        and len({i.intencao_id for i in intencoes}) != len(intencoes)
    ):
        raise ValueError("intencao_id duplicado")
    preparar = (
        _preparar_par_compacto
        if _lotes_compactos(intencoes) is not None else _preparar_par
    )
    simbolos, precalculo, eventos = preparar(
        intencoes, carregador, calendario, params_normal.filtro_janela_pregoes,
    )
    internos = dict(
        somente_series=somente_series,
        _precalculo=precalculo,
        _somente_series=somente_series,
        _eventos_preparados=eventos if somente_series else None,
        _intencoes_validadas=True,
        _simbolos_preparados=simbolos,
    )
    custos_pessimistas_nulos = all(
        not barras or all(b.spread == barras[0].spread for b in barras)
        for barras in precalculo.barras_por_simbolo.values()
    ) and all(
        not _especificacao(
            contratos, simbolo, precalculo.primeiro_instante[simbolo],
        ).dividendo_pessimista_usd_por_lote_dia
        for simbolo in simbolos
    )
    normal = simular(
        intencoes, carregador, contratos, calendario, params_normal, **internos,
    )
    if custos_pessimistas_nulos:
        return normal, normal
    pessimista = simular(
        intencoes, carregador, contratos, calendario, params_pessimista, **internos,
    )
    return normal, pessimista


_ChaveIsolada = TypeVar("_ChaveIsolada", bound=Hashable)


class _ArvoreExtremos:
    """Indice imutavel para achar a primeira barra que toca stop ou alvo."""

    __slots__ = ("n", "min_bid", "max_bid", "min_ask", "max_ask")

    def __init__(
        self, barras: Sequence[BarraMotor], spreads: Sequence[int]
    ) -> None:
        n = 1
        while n < len(barras):
            n *= 2
        self.n = n
        self.min_bid = [float("inf")] * (2 * n)
        self.max_bid = [float("-inf")] * (2 * n)
        self.min_ask = [float("inf")] * (2 * n)
        self.max_ask = [float("-inf")] * (2 * n)
        for numero, (b, spread) in enumerate(zip(barras, spreads, strict=True)):
            folha = n + numero
            self.min_bid[folha] = b.minima
            self.max_bid[folha] = b.maxima
            self.min_ask[folha] = b.minima + spread
            self.max_ask[folha] = b.maxima + spread
        for no in range(n - 1, 0, -1):
            esquerdo, direito = 2 * no, 2 * no + 1
            self.min_bid[no] = min(self.min_bid[esquerdo], self.min_bid[direito])
            self.max_bid[no] = max(self.max_bid[esquerdo], self.max_bid[direito])
            self.min_ask[no] = min(self.min_ask[esquerdo], self.min_ask[direito])
            self.max_ask[no] = max(self.max_ask[esquerdo], self.max_ask[direito])

    def primeiro(
        self, inicio: int, fim: int, lado: Lado, stop: int,
        alvo: int | None,
    ) -> int | None:
        def toca(no: int) -> bool:
            if lado is Lado.COMPRA:
                return self.min_bid[no] <= stop or (
                    alvo is not None and self.max_bid[no] >= alvo
                )
            return self.max_ask[no] >= stop or (
                alvo is not None and self.min_ask[no] <= alvo
            )

        def buscar(no: int, esquerda: int, direita: int) -> int | None:
            if direita <= inicio or esquerda >= fim or not toca(no):
                return None
            if direita - esquerda == 1:
                return esquerda
            meio = (esquerda + direita) // 2
            achado = buscar(2 * no, esquerda, meio)
            return achado if achado is not None else buscar(
                2 * no + 1, meio, direita
            )

        return buscar(1, 0, self.n) if inicio < fim else None


def _eventos_dirigidos(
    intencoes: Sequence[Intencao],
    barras_evento: Mapping[datetime, Sequence[BarraMotor]],
    linhas: Mapping[str, Sequence[BarraMotor]],
    tempos: Mapping[str, Sequence[datetime]],
    arvores: Mapping[str, _ArvoreExtremos],
    indices: Mapping[str, _IndiceBarras],
    indicadores: Mapping[tuple[str, str, int, date], _IndicadorHora],
    specs: Mapping[str, EspecificacaoAtivo],
    calendario: Any,
    params: ParamsMotor,
) -> dict[datetime, tuple[BarraMotor, ...]]:
    """Seleciona somente barras capazes de alterar uma simulacao isolada.

    A entrada e localizada por indice; a busca subsequente para no primeiro
    stop, alvo ou limite. Barras intermediarias so entram se carregarem ajuste
    contratual. O simulador comum continua sendo o executor e o oraculo das
    regras economicas.
    """
    escolhidos: set[datetime] = set()

    for i in intencoes:
        entrada_barra, _ = _selecionar_entrada(i, indices[i.simbolo])
        if entrada_barra is None:
            continue
        escolhidos.add(entrada_barra.inicio)
        try:
            sessao = _sessao(
                calendario, i.simbolo, entrada_barra.data_servidor
            )
        except (KeyError, TypeError):
            continue
        limite = min(
            i.hora_limite,
            sessao.limite_zeragem_utc or i.hora_limite,
            sessao.virada_utc,
        )
        spec = specs[i.simbolo]
        spread_entrada = _spread_execucao(
            indicadores, entrada_barra, params, calendario
        )
        preco_entrada = _preco_entrada(i, entrada_barra, spread_entrada, spec)
        stop = (
            preco_entrada - i.stop_pontos
            if i.lado is Lado.COMPRA else preco_entrada + i.stop_pontos
        )
        alvo = None if i.alvo_pontos is None else (
            preco_entrada + i.alvo_pontos
            if i.lado is Lado.COMPRA else preco_entrada - i.alvo_pontos
        )
        barras = linhas.get(i.simbolo, ())
        instantes = tempos.get(i.simbolo, ())
        inicio = bisect_left(instantes, entrada_barra.inicio)
        fronteira = bisect_left(instantes, limite, lo=inicio)
        gatilho = arvores[i.simbolo].primeiro(
            inicio, fronteira, i.lado, stop, alvo
        )
        fim_posicao = gatilho if gatilho is not None else fronteira
        if fim_posicao < len(barras):
            escolhidos.add(barras[fim_posicao].inicio)
        elif barras:
            escolhidos.add(barras[-1].inicio)
        for ajuste in spec.ajustes:
            numero = bisect_left(instantes, ajuste.instante, lo=inicio)
            if numero <= fim_posicao and numero < len(barras):
                if barras[numero].inicio == ajuste.instante:
                    escolhidos.add(ajuste.instante)

    return {
        instante: tuple(barras_evento[instante])
        for instante in sorted(escolhidos)
        if instante in barras_evento
    }


@dataclass(frozen=True, slots=True)
class _ExecucaoIsolada:
    dia: date
    risco_usd: float
    margem_usd: float
    pnl_usd: float
    saida_em: datetime
    saida_dentro_barra: bool
    notional_usd: float
    lado: Lado
    bloco: str


_FALLBACK_EXECUCAO_ISOLADA = object()


def _simular_isolada_series_rapida(
    intencoes: Sequence[Intencao],
    indices: Mapping[str, _IndiceBarras],
    linhas: Mapping[str, Sequence[BarraMotor]],
    tempos: Mapping[str, Sequence[datetime]],
    arvores: Mapping[str, _ArvoreExtremos],
    indicadores: Mapping[tuple[str, str, int, date], _IndicadorHora],
    specs: Mapping[str, EspecificacaoAtivo],
    datas_base: Mapping[str, frozenset[date]],
    calendario: Any,
    params: ParamsMotor,
    entradas_compartilhadas: Mapping[
        tuple[str, Campanha, datetime], tuple[BarraMotor | None, bool]
    ] | None = None,
    execucoes_compartilhadas: dict[tuple[Any, ...], Any] | None = None,
    sessoes_compartilhadas: dict[tuple[str, date], Any] | None = None,
    indicadores_compartilhados: dict[tuple[str, datetime], _IndicadorHora] | None = None,
    spreads_compartilhados: dict[tuple[str, datetime], int] | None = None,
) -> ResultadoMotor | None:
    """Executor O(eventos log barras) para uma regra day-trade isolada."""
    simbolos = {i.simbolo for i in intencoes}
    if (
        not params.capital_constante or len(simbolos) != 1
        or any(i.campanha is not Campanha.DAY_TRADE_M15 for i in intencoes)
    ):
        return None
    simbolo = next(iter(simbolos), None)
    if simbolo is None:
        return None
    # O simulador comum ordena admissões da mesma abertura por peso e id. A
    # chave B3 pode ter dezenas de sinais por dia; abandonar o caminho rápido
    # nesse caso era o gargalo dominante das séries isoladas.
    spec = specs[simbolo]
    pnl_dia: defaultdict[date, float] = defaultdict(float)
    expo_compra: defaultdict[date, float] = defaultdict(float)
    expo_venda: defaultdict[date, float] = defaultdict(float)
    realizado: defaultdict[date, float] = defaultdict(float)
    bloqueados: set[date] = set()
    blocos: list[str] = []
    barras = linhas[simbolo]
    instantes = tempos[simbolo]
    ocupado_ate: datetime | None = None
    # Stop/alvo acontece dentro da barra: uma intenção admitida na abertura da
    # mesma barra ainda encontra a posição aberta. Zeragem acontece na abertura
    # da fronteira e libera o símbolo antes das admissões daquele instante.
    ocupado_inclusive = False
    cache_execucao = execucoes_compartilhadas if execucoes_compartilhadas is not None else {}
    cache_sessoes = sessoes_compartilhadas if sessoes_compartilhadas is not None else {}
    cache_indicadores = indicadores_compartilhados if indicadores_compartilhados is not None else {}
    cache_spreads = spreads_compartilhados if spreads_compartilhados is not None else {}

    def sessao_rapida(dia: date):
        chave = (simbolo, dia)
        if chave not in cache_sessoes:
            cache_sessoes[chave] = _sessao(calendario, simbolo, dia)
        return cache_sessoes[chave]

    def indicador_rapido(barra: BarraMotor) -> _IndicadorHora:
        chave = (simbolo, barra.inicio)
        if chave not in cache_indicadores:
            cache_indicadores[chave] = _indicador_hora(indicadores, barra, calendario)
        return cache_indicadores[chave]

    def spread_rapido(barra: BarraMotor) -> int:
        chave = (simbolo, barra.inicio)
        if chave not in cache_spreads:
            cache_spreads[chave] = _spread_execucao(indicadores, barra, params, calendario)
        return cache_spreads[chave]

    def preparar_execucao(
        i: Intencao, b: BarraMotor, realizado_antes: float = 0.0,
    ):
        entrada_prevista = _ceil_m15(i.barra_sinal_fim)
        if entrada_prevista >= i.hora_limite or b.inicio >= i.hora_limite:
            return None
        dia = b.data_servidor
        try:
            sessao = sessao_rapida(dia)
        except (KeyError, TypeError):
            return None
        limite = min(i.hora_limite, sessao.limite_zeragem_utc or i.hora_limite, sessao.virada_utc)
        if b.inicio >= limite:
            return None
        indicador = indicador_rapido(b)
        if (indicador.spread_mediano is None or indicador.movimento_mediano is None
                or indicador.spread_mediano > params.filtro_spread_mov * indicador.movimento_mediano):
            return None
        risco_lote = ((i.stop_pontos + spec.slippage_pontos)
                      * spec.valor_ponto_moeda_cotacao * b.cambio_usd)
        quantidade = _passo_quantidade(params.risco_por_operacao * params.capital_usd / risco_lote, spec)
        if quantidade < spec.lote_minimo:
            return None
        risco_novo = risco_lote * quantidade
        piso = (1.0 - params.limiar_m) * params.capital_usd
        alvo_risco = params.risco_aberto_alvo * params.limiar_m * params.capital_usd
        limite_classe = params.fracao_um_terco * params.limiar_m * params.capital_usd
        if (risco_novo > alvo_risco + 1e-9
                or params.capital_usd - risco_novo < piso - 1e-9
                or (spec.limita_um_terco and risco_novo > limite_classe + 1e-9)):
            return None
        spread_entrada = spread_rapido(b)
        preco_entrada = _preco_entrada(i, b, spread_entrada, spec)
        margem = spec.margem_no_instante(
            b.inicio, preco_entrada, b.cambio_usd
        ) * quantidade
        stop = preco_entrada - i.stop_pontos if i.lado is Lado.COMPRA else preco_entrada + i.stop_pontos
        alvo = None if i.alvo_pontos is None else (
            preco_entrada + i.alvo_pontos if i.lado is Lado.COMPRA else preco_entrada - i.alvo_pontos
        )
        efetiva = _com_hora_limite(i, limite)
        posicao = _Posicao(
            efetiva, spec, quantidade, b.inicio, preco_entrada, stop, alvo,
            spread_entrada, risco_novo, margem, dia.isoformat(), False,
            ultimo_bid=b.abertura, ultimo_spread=spread_entrada,
            ultimo_spread_real=b.spread, ultimo_cambio_usd=b.cambio_usd,
        )
        if (params.pessimista and spec.limita_um_terco
                and spec.dividendo_pessimista_usd_por_lote_dia
                and not any(a.tipo == "dividendo" and a.instante.date() == dia for a in spec.ajustes)):
            posicao.ajustes_usd -= spec.dividendo_pessimista_usd_por_lote_dia * quantidade
        numero_inicio = bisect_left(instantes, b.inicio)
        fronteira = bisect_left(instantes, limite, lo=numero_inicio)
        atingimento = None
        gatilho = arvores[simbolo].primeiro(
            numero_inicio, fronteira, i.lado, stop, alvo
        )
        if gatilho is not None:
            saida_barra = barras[gatilho]
            spread_saida = spread_rapido(saida_barra)
            if atingimento is None:
                barra_custo = (saida_barra if spread_saida == saida_barra.spread
                               else replace(saida_barra, spread=spread_saida))
                atingimento = _atingimento(posicao, barra_custo, spread_real=saida_barra.spread)
            if atingimento is None:
                return _FALLBACK_EXECUCAO_ISOLADA
            preco_saida, cambio_saida = atingimento[1], saida_barra.cambio_usd
            instante_saida, saida_dentro_barra = saida_barra.inicio, True
        elif fronteira < len(barras):
            saida_barra = barras[fronteira]
            spread_saida = spread_rapido(saida_barra)
            preco_saida = _preco_saida(
                posicao, saida_barra.abertura, spread_saida, slippage=True,
                spread_real=saida_barra.spread,
            )
            cambio_saida = saida_barra.cambio_usd
            instante_saida, saida_dentro_barra = saida_barra.inicio, False
        else:
            saida_barra = barras[-1]
            spread_saida = spread_rapido(saida_barra)
            delta = spec.slippage_pontos + max(0, spread_saida - saida_barra.spread)
            preco_saida = stop - delta if i.lado is Lado.COMPRA else stop + delta
            cambio_saida = saida_barra.cambio_usd
            instante_saida, saida_dentro_barra = i.hora_limite, False

        # No modo de taxa, a margem muda junto com preço e câmbio. O executor
        # comum confere C8 na abertura e no fechamento de toda barra enquanto
        # a posição está aberta. Reproduzimos apenas esses pontos de decisão;
        # barras sem posição continuam ausentes do relógio dirigido por eventos.
        if spec.taxa_margem_notional is not None:
            fim_natural = gatilho if gatilho is not None else fronteira

            def margem_excede(bid: int, spread: int, cambio: float) -> bool:
                posicao.ultimo_bid = bid
                posicao.ultimo_spread = spread
                posicao.ultimo_spread_real = barra_margem.spread
                posicao.ultimo_cambio_usd = cambio
                usada = spec.margem_no_instante(
                    barra_margem.inicio, bid, cambio,
                ) * quantidade
                risco = _risco_atual(posicao, bid, spread, cambio)
                aberto = _pnl(
                    posicao,
                    _preco_saida(
                        posicao, bid, spread,
                        spread_real=barra_margem.spread,
                    ),
                    cambio,
                )
                estressado = (
                    params.capital_usd + realizado_antes + aberto - risco
                )
                fracao = usada / estressado if estressado > 0 else float("inf")
                return fracao >= params.margem_max

            for numero in range(numero_inicio, min(fim_natural + 1, len(barras))):
                barra_margem = barras[numero]
                spread_margem = spread_rapido(barra_margem)
                # Na fronteira, a zeragem acontece antes do controle de margem.
                if numero > numero_inicio and numero < fronteira and margem_excede(
                    barra_margem.abertura, spread_margem,
                    barra_margem.cambio_usd,
                ):
                    preco_saida = _preco_saida(
                        posicao, barra_margem.abertura, spread_margem,
                        slippage=True, spread_real=barra_margem.spread,
                    )
                    cambio_saida = barra_margem.cambio_usd
                    instante_saida = barra_margem.inicio
                    saida_dentro_barra = False
                    break
                # Stop/alvo dentro da barra precede o controle posterior.
                if gatilho is not None and numero == gatilho:
                    break
                if numero < fronteira and margem_excede(
                    barra_margem.fechamento, spread_margem,
                    barra_margem.cambio_usd,
                ):
                    preco_saida = _preco_saida(
                        posicao, barra_margem.fechamento, spread_margem,
                        slippage=True, spread_real=barra_margem.spread,
                    )
                    cambio_saida = barra_margem.cambio_usd
                    instante_saida = barra_margem.inicio
                    saida_dentro_barra = True
                    break
        for ajuste in spec.ajustes:
            if not (b.inicio <= ajuste.instante <= instante_saida):
                continue
            numero_ajuste = bisect_left(instantes, ajuste.instante, lo=numero_inicio)
            if numero_ajuste >= len(instantes) or instantes[numero_ajuste] != ajuste.instante:
                continue
            dividendo_valido = ajuste.tipo != "dividendo" or spec.modo_contrato in {
                "spot", "indice_spot", "acao"
            }
            if dividendo_valido and ajuste.tipo != "swap":
                sinal = 1.0 if i.lado is Lado.COMPRA else -1.0
                posicao.ajustes_usd += sinal * ajuste.valor_por_lote * ajuste.cambio_usd * quantidade
        valor = _pnl(posicao, preco_saida, cambio_saida)
        notional = abs(preco_entrada * spec.valor_ponto_moeda_cotacao * quantidade * b.cambio_usd)
        return _ExecucaoIsolada(
            dia, risco_novo, margem, valor, instante_saida, saida_dentro_barra,
            notional, i.lado, dia.isoformat(),
        )

    # As listas chegam ordenadas por abertura. Depois de uma admissão NETTING,
    # nenhuma intenção cuja abertura caia dentro do intervalo ocupado pode
    # mudar a série. B3 produz cerca de 45 delas por dia; saltá-las por busca
    # binária é o ganho dominante das séries isoladas densas.
    entradas_previstas = [_ceil_m15(i.barra_sinal_fim) for i in intencoes]
    numero_intencao = 0
    while numero_intencao < len(intencoes):
        if ocupado_ate is not None:
            busca = bisect_right if ocupado_inclusive else bisect_left
            numero_intencao = busca(
                entradas_previstas, ocupado_ate, lo=numero_intencao
            )
            if numero_intencao >= len(intencoes):
                break
        i = intencoes[numero_intencao]
        numero_intencao += 1
        chave_entrada = (i.simbolo, i.campanha, i.barra_sinal_fim)
        selecao = entradas_compartilhadas.get(chave_entrada) if entradas_compartilhadas is not None else None
        if selecao is None:
            selecao = _selecionar_entrada(i, indices[simbolo])
            if isinstance(entradas_compartilhadas, dict):
                entradas_compartilhadas[chave_entrada] = selecao
        b, _ = selecao
        if b is None:
            continue
        if ocupado_ate is not None and (
            b.inicio < ocupado_ate or (ocupado_inclusive and b.inicio == ocupado_ate)
        ):
            continue
        chave_execucao = (
            simbolo, b.inicio, i.lado, i.stop_pontos,
            i.alvo_pontos, i.hora_limite,
        )
        if spec.taxa_margem_notional is None and chave_execucao in cache_execucao:
            execucao = cache_execucao[chave_execucao]
        else:
            execucao = preparar_execucao(
                i, b, realizado[b.data_servidor],
            )
            if spec.taxa_margem_notional is None:
                cache_execucao[chave_execucao] = execucao
        if execucao is _FALLBACK_EXECUCAO_ISOLADA:
            return None
        if execucao is None:
            continue
        dia = execucao.dia
        if dia in bloqueados:
            continue
        limite_dia = params.limite_perda_dia * params.capital_usd
        if max(0.0, -realizado[dia]) + execucao.risco_usd >= limite_dia - 1e-12:
            continue
        estressado = params.capital_usd + realizado[dia] - execucao.risco_usd
        fracao = execucao.margem_usd / estressado if estressado > 0 else float("inf")
        if fracao > params.margem_max + 1e-12:
            continue
        ocupado_ate = execucao.saida_em
        ocupado_inclusive = execucao.saida_dentro_barra
        pnl_dia[dia] += execucao.pnl_usd
        realizado[dia] += execucao.pnl_usd
        blocos.append(execucao.bloco)
        (expo_compra if execucao.lado is Lado.COMPRA else expo_venda)[dia] += execucao.notional_usd
        if max(0.0, -realizado[dia]) > limite_dia + 1e-12:
            bloqueados.add(dia)
        continue
        entrada_prevista = _ceil_m15(i.barra_sinal_fim)
        if entrada_prevista >= i.hora_limite or b.inicio >= i.hora_limite:
            continue
        dia = b.data_servidor
        if dia in bloqueados:
            continue
        try:
            sessao = sessao_rapida(dia)
        except (KeyError, TypeError):
            continue
        limite = min(
            i.hora_limite, sessao.limite_zeragem_utc or i.hora_limite,
            sessao.virada_utc,
        )
        if b.inicio >= limite:
            continue
        indicador = indicador_rapido(b)
        if (
            indicador.spread_mediano is None
            or indicador.movimento_mediano is None
            or indicador.spread_mediano
            > params.filtro_spread_mov * indicador.movimento_mediano
        ):
            continue
        risco_lote = (
            (i.stop_pontos + spec.slippage_pontos)
            * spec.valor_ponto_moeda_cotacao * b.cambio_usd
        )
        quantidade = _passo_quantidade(
            params.risco_por_operacao * params.capital_usd / risco_lote, spec
        )
        if quantidade < spec.lote_minimo:
            continue
        risco_novo = risco_lote * quantidade
        piso = (1.0 - params.limiar_m) * params.capital_usd
        alvo_risco = (
            params.risco_aberto_alvo * params.limiar_m * params.capital_usd
        )
        limite_classe = (
            params.fracao_um_terco * params.limiar_m * params.capital_usd
        )
        if (
            risco_novo > alvo_risco + 1e-9
            or params.capital_usd - risco_novo < piso - 1e-9
            or (spec.limita_um_terco and risco_novo > limite_classe + 1e-9)
        ):
            continue
        limite_dia = params.limite_perda_dia * params.capital_usd
        if max(0.0, -realizado[dia]) + risco_novo >= limite_dia - 1e-12:
            continue
        spread_entrada = spread_rapido(b)
        preco_entrada = _preco_entrada(i, b, spread_entrada, spec)
        margem = spec.margem_no_instante(
            b.inicio, preco_entrada, b.cambio_usd
        ) * quantidade
        estressado = params.capital_usd + realizado[dia] - risco_novo
        fracao = margem / estressado if estressado > 0 else float("inf")
        if fracao > params.margem_max + 1e-12:
            continue

        stop = (
            preco_entrada - i.stop_pontos
            if i.lado is Lado.COMPRA else preco_entrada + i.stop_pontos
        )
        alvo = None if i.alvo_pontos is None else (
            preco_entrada + i.alvo_pontos
            if i.lado is Lado.COMPRA else preco_entrada - i.alvo_pontos
        )
        efetiva = _com_hora_limite(i, limite)
        posicao = _Posicao(
            efetiva, spec, quantidade, b.inicio, preco_entrada, stop, alvo,
            spread_entrada, risco_novo, margem, dia.isoformat(), False,
            ultimo_bid=b.abertura, ultimo_spread=spread_entrada,
            ultimo_spread_real=b.spread, ultimo_cambio_usd=b.cambio_usd,
        )
        if (
            params.pessimista and spec.limita_um_terco
            and spec.dividendo_pessimista_usd_por_lote_dia
            and not any(
                a.tipo == "dividendo" and a.instante.date() == dia
                for a in spec.ajustes
            )
        ):
            posicao.ajustes_usd -= (
                spec.dividendo_pessimista_usd_por_lote_dia * quantidade
            )
        inicio = bisect_left(instantes, b.inicio)
        fronteira = bisect_left(instantes, limite, lo=inicio)
        gatilho = arvores[simbolo].primeiro(
            inicio, fronteira, i.lado, stop, alvo
        )
        if gatilho is not None:
            saida_barra = barras[gatilho]
            spread_saida = spread_rapido(saida_barra)
            barra_custo = (
                saida_barra if spread_saida == saida_barra.spread
                else replace(saida_barra, spread=spread_saida)
            )
            atingimento = _atingimento(
                posicao, barra_custo, spread_real=saida_barra.spread
            )
            if atingimento is None:
                return None
            preco_saida = atingimento[1]
            cambio_saida = saida_barra.cambio_usd
            instante_saida = saida_barra.inicio
            saida_dentro_barra = True
        elif fronteira < len(barras):
            saida_barra = barras[fronteira]
            spread_saida = spread_rapido(saida_barra)
            preco_saida = _preco_saida(
                posicao, saida_barra.abertura, spread_saida, slippage=True,
                spread_real=saida_barra.spread,
            )
            cambio_saida = saida_barra.cambio_usd
            instante_saida = saida_barra.inicio
            saida_dentro_barra = False
        else:
            saida_barra = barras[-1]
            spread_saida = spread_rapido(saida_barra)
            delta = spec.slippage_pontos + max(
                0, spread_saida - saida_barra.spread
            )
            preco_saida = stop - delta if i.lado is Lado.COMPRA else stop + delta
            cambio_saida = saida_barra.cambio_usd
            instante_saida = i.hora_limite
            saida_dentro_barra = False
        for ajuste in spec.ajustes:
            if not (b.inicio <= ajuste.instante <= instante_saida):
                continue
            numero_ajuste = bisect_left(instantes, ajuste.instante, lo=inicio)
            if (
                numero_ajuste >= len(instantes)
                or instantes[numero_ajuste] != ajuste.instante
            ):
                continue
            dividendo_valido = ajuste.tipo != "dividendo" or spec.modo_contrato in {
                "spot", "indice_spot", "acao"
            }
            swap_valido = ajuste.tipo != "swap"  # day trade nunca paga swap
            if dividendo_valido and swap_valido:
                sinal = 1.0 if i.lado is Lado.COMPRA else -1.0
                posicao.ajustes_usd += (
                    sinal * ajuste.valor_por_lote * ajuste.cambio_usd * quantidade
                )
        valor = _pnl(posicao, preco_saida, cambio_saida)
        ocupado_ate = instante_saida
        ocupado_inclusive = saida_dentro_barra
        pnl_dia[dia] += valor
        realizado[dia] += valor
        blocos.append(dia.isoformat())
        notional = abs(
            preco_entrada * spec.valor_ponto_moeda_cotacao
            * quantidade * b.cambio_usd
        )
        cache_execucao[chave_execucao] = _ExecucaoIsolada(
            dia, risco_novo, margem, valor, instante_saida,
            saida_dentro_barra, notional, i.lado, dia.isoformat(),
        )
        (expo_compra if i.lado is Lado.COMPRA else expo_venda)[dia] += notional
        if max(0.0, -realizado[dia]) > limite_dia + 1e-12:
            bloqueados.add(dia)

    inicio_prova = min(
        (i.barra_sinal_fim.date() for i in intencoes), default=date.max
    )
    datas = tuple(sorted(d for d in datas_base[simbolo] if d >= inicio_prova))
    pnl = tuple(float(pnl_dia[d]) for d in datas)
    return ResultadoMotor(
        datas, pnl, tuple(x / params.capital_usd for x in pnl),
        tuple(float(expo_compra[d]) for d in datas),
        tuple(float(expo_venda[d]) for d in datas), tuple(blocos), (), (), (), (),
    )


def simular_isoladas(
    intencoes_por_chave: Mapping[
        _ChaveIsolada, Mapping[datetime, Sequence[Intencao]] | Iterable[Intencao]
    ],
    carregador: Any,
    contratos: Any,
    calendario: Any,
    params: ParamsMotor,
    *,
    somente_series: bool = False,
    lote_continuo_diagnostico: bool = False,
) -> dict[_ChaveIsolada, ResultadoMotor]:
    """Simula chaves isoladas compartilhando o pre-calculo causal por ativo.

    Cada resultado continua sendo uma simulacao independente. Chaves com o
    mesmo conjunto de simbolos reutilizam barras, indices e indicadores de
    hora; nenhuma posicao, recusa ou estado atravessa de uma chave para outra.
    """
    listas: dict[_ChaveIsolada, Sequence[Any]] = {}
    grupos: defaultdict[tuple[str, ...], list[_ChaveIsolada]] = defaultdict(list)
    for chave, origem in intencoes_por_chave.items():
        if isinstance(origem, Mapping):
            itens = [i for instante in sorted(origem) for i in origem[instante]]
        elif eh_sequencia_compacta(origem):
            itens = origem
        else:
            itens = list(origem)
        if somente_series and itens and all(
            i.campanha is Campanha.DAY_TRADE_M15 for i in itens
        ):
            peso = itens[0].peso_admissao
            anterior_sinal = itens[0].barra_sinal_fim
            simples = anterior_sinal.minute % 15 == 0
            for i in itens[1:]:
                sinal = i.barra_sinal_fim
                if (i.peso_admissao != peso or sinal.minute % 15
                        or sinal < anterior_sinal):
                    simples = False
                    break
                anterior_sinal = sinal
            if not simples:
                chave_ordem = lambda i: (
                    _ceil_m15(i.barra_sinal_fim),
                    -i.peso_admissao, i.intencao_id,
                )
                if isinstance(itens, LoteIntencoesCompactas):
                    ordem = np.asarray(
                        sorted(range(len(itens)), key=lambda k: chave_ordem(itens[k])),
                        dtype=np.int64,
                    )
                    itens = SelecaoIntencoesCompactas(itens, ordem)
                else:
                    itens = sorted(itens, key=chave_ordem)
        listas[chave] = itens
        grupos[tuple(sorted({i.simbolo for i in itens}))].append(chave)

    saida: dict[_ChaveIsolada, ResultadoMotor] = {}
    for simbolos, chaves in grupos.items():
        if not simbolos:
            for chave in chaves:
                saida[chave] = simular(
                    listas[chave], carregador, contratos, calendario, params,
                    lote_continuo_diagnostico=lote_continuo_diagnostico,
                )
            continue

        barras_por_simbolo = {s: _barras_do(carregador, s) for s in simbolos}
        indices = {s: _indexar_barras(bs) for s, bs in barras_por_simbolo.items()}
        grupo_rapido = (
            somente_series
            and not lote_continuo_diagnostico
            and params.capital_constante
            and len(simbolos) == 1
            and all(
                all(i.campanha is Campanha.DAY_TRADE_M15 for i in listas[chave])
                for chave in chaves
            )
        )
        if grupo_rapido:
            # Não materializa o mapa global de eventos nem o reconstrói em
            # ordem temporal. Para M15 isolado, basta a última barra de cada
            # instante, exatamente como o ``por_simbolo`` do caminho genérico.
            linhas_rapidas: dict[str, tuple[BarraMotor, ...]] = {}
            for s, bs in barras_por_simbolo.items():
                por_inicio = {
                    b.inicio: b for b in bs if b.tempo_grafico != "D1"
                }
                linhas_rapidas[s] = tuple(
                    por_inicio[t] for t in sorted(por_inicio)
                )
            tempos_rapidos = {
                s: tuple(b.inicio for b in bs)
                for s, bs in linhas_rapidas.items()
            }
            indicadores_rapidos = _precalcular_horas(
                barras_por_simbolo, calendario,
                params.filtro_janela_pregoes, None,
            )
            arvores_rapidas = {
                s: _ArvoreExtremos(
                    bs,
                    tuple(
                        _spread_execucao(
                            indicadores_rapidos, b, params, calendario
                        )
                        for b in bs
                    ),
                )
                for s, bs in linhas_rapidas.items()
            }
            specs_rapidas = {
                s: _especificacao(
                    contratos, s,
                    min((b.inicio for b in barras_por_simbolo[s]),
                        default=datetime.now(UTC)),
                )
                for s in simbolos
            }
            datas_rapidas = {
                s: frozenset(b.data_servidor for b in bs)
                for s, bs in barras_por_simbolo.items()
            }
            entradas_rapidas: dict[
                tuple[str, Campanha, datetime], tuple[BarraMotor | None, bool]
            ] = {}
            execucoes_rapidas: dict[tuple[Any, ...], Any] = {}
            sessoes_rapidas: dict[tuple[str, date], Any] = {}
            indicadores_consultados: dict[
                tuple[str, datetime], _IndicadorHora
            ] = {}
            spreads_rapidos: dict[tuple[str, datetime], int] = {}
            falhas: list[_ChaveIsolada] = []
            for chave in chaves:
                rapido = _simular_isolada_series_rapida(
                    listas[chave], indices, linhas_rapidas, tempos_rapidos,
                    arvores_rapidas, indicadores_rapidos, specs_rapidas,
                    datas_rapidas, calendario, params, entradas_rapidas,
                    execucoes_rapidas, sessoes_rapidas,
                    indicadores_consultados, spreads_rapidos,
                )
                if rapido is None:
                    falhas.append(chave)
                else:
                    saida[chave] = rapido
            if not falhas:
                continue
            chaves = falhas
        barras_evento_mut: dict[datetime, list[BarraMotor]] = defaultdict(list)
        for barras in barras_por_simbolo.values():
            dias_com_intradia = {
                b.data_servidor for b in barras if b.tempo_grafico in {"M15", "H1"}
            }
            for b in barras:
                if b.tempo_grafico != "D1" or b.data_servidor not in dias_com_intradia:
                    barras_evento_mut[b.inicio].append(b)

        horas_entrada: set[tuple[str, str, int]] = set()
        entradas_compartilhadas: dict[
            tuple[str, Campanha, datetime], tuple[BarraMotor | None, bool]
        ] = {}
        if not somente_series:
            for chave in chaves:
                for intencao in listas[chave]:
                    chave_entrada = (
                        intencao.simbolo, intencao.campanha, intencao.barra_sinal_fim
                    )
                    selecao = entradas_compartilhadas.get(chave_entrada)
                    if selecao is None:
                        selecao = _selecionar_entrada(
                            intencao, indices[intencao.simbolo]
                        )
                        entradas_compartilhadas[chave_entrada] = selecao
                    barra, _ = selecao
                    if barra is not None:
                        horas_entrada.add((
                            barra.simbolo, barra.tempo_grafico,
                            _hora_servidor(barra, calendario),
                        ))
        indicadores = _precalcular_horas(
            barras_por_simbolo,
            calendario,
            params.filtro_janela_pregoes,
            None if params.pessimista or somente_series else horas_entrada,
        )
        barras_evento = {
            instante: tuple(bs) for instante, bs in barras_evento_mut.items()
        }
        linhas_evento: defaultdict[str, list[BarraMotor]] = defaultdict(list)
        for instante in sorted(barras_evento):
            por_simbolo = {b.simbolo: b for b in barras_evento[instante]}
            for simbolo, barra in por_simbolo.items():
                linhas_evento[simbolo].append(barra)
        tempos_evento = {
            s: tuple(b.inicio for b in bs) for s, bs in linhas_evento.items()
        }
        arvores_evento = {
            s: _ArvoreExtremos(
                bs,
                tuple(
                    _spread_execucao(indicadores, b, params, calendario)
                    for b in bs
                ),
            )
            for s, bs in linhas_evento.items()
        }
        precalculo = _PrecalculoCompartilhado(
            barras_por_simbolo,
            indices,
            barras_evento,
            indicadores,
            {s: bs[0].inicio for s, bs in barras_por_simbolo.items() if bs},
            {
                s: frozenset(b.data_servidor for b in bs)
                for s, bs in barras_por_simbolo.items()
            },
        )
        specs = {
            s: _especificacao(
                contratos, s,
                min((b.inicio for b in barras_por_simbolo[s]), default=datetime.now(UTC)),
            )
            for s in simbolos
        }
        execucoes_compartilhadas: dict[tuple[Any, ...], Any] = {}
        sessoes_compartilhadas: dict[tuple[str, date], Any] = {}
        indicadores_compartilhados: dict[tuple[str, datetime], _IndicadorHora] = {}
        spreads_compartilhados: dict[tuple[str, datetime], int] = {}
        for chave in chaves:
            if somente_series and not lote_continuo_diagnostico:
                rapido = _simular_isolada_series_rapida(
                    listas[chave], indices, linhas_evento, tempos_evento,
                    arvores_evento, indicadores, specs, precalculo.datas,
                    calendario, params, entradas_compartilhadas,
                    execucoes_compartilhadas,
                    sessoes_compartilhadas, indicadores_compartilhados,
                    spreads_compartilhados,
                )
                if rapido is not None:
                    saida[chave] = rapido
                    continue
            precalculo_chave = precalculo
            if somente_series:
                precalculo_chave = replace(
                    precalculo,
                    barras_evento=_eventos_dirigidos(
                        listas[chave], barras_evento, linhas_evento,
                        tempos_evento, arvores_evento, indices, indicadores,
                        specs, calendario, params,
                    ),
                )
            saida[chave] = simular(
                listas[chave], carregador, contratos, calendario, params,
                lote_continuo_diagnostico=lote_continuo_diagnostico,
                _precalculo=precalculo_chave,
                _somente_series=somente_series,
            )
    return saida


_CAMPOS_CONFERENCIA = (
    "evento", "intencao_id", "preco_entrada", "stop", "risco_usd", "margem_usd"
)


def conferir_livro(
    resultado: ResultadoMotor,
    intencoes_por_barra: Mapping[datetime, Sequence[Intencao]] | Iterable[Intencao],
    carregador: Any,
    contratos: Any,
    calendario: Any,
    params: ParamsMotor,
) -> None:
    """F2-5: recompõe e exige igualdade de entrada, stop, risco e margem."""
    esperado = simular(intencoes_por_barra, carregador, contratos, calendario, params)
    reais = [r for r in resultado.livro if r.get("evento") == "abertura"]
    exp = [r for r in esperado.livro if r.get("evento") == "abertura"]
    if len(reais) != len(exp):
        raise ValueError("quantidade de aberturas diverge")
    for numero, (r, e) in enumerate(zip(reais, exp, strict=True)):
        for campo in _CAMPOS_CONFERENCIA:
            if r.get(campo) != e.get(campo):
                raise ValueError(f"livro diverge na abertura {numero}, campo {campo}")


__all__ = [
    "AjusteMotor", "BarraMotor", "EspecificacaoAtivo", "conferir_livro", "simular",
    "simular_isoladas",
]
