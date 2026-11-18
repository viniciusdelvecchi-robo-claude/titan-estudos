"""F2-10: viabilidade do sistema completo em mundos sintéticos.

Este módulo não escolhe o risco aberto alvo. A seção 4 manda calibrá-lo como o
maior valor cuja revogação em três anos, em mundos sem vantagem, tenha limite
superior CP95% de no máximo 10%. Por isso ``risco_aberto_por_m`` é uma entrada
obrigatória, já produzida por essa calibração.
"""

from __future__ import annotations

from bisect import bisect_right
from calendar import monthrange
from dataclasses import dataclass, replace
from datetime import date, datetime
import math
from statistics import median, stdev
from typing import Mapping, Sequence

from titan.motor import simular
from titan.sintetico import Mundo
from titan.tipos_f2 import EstadoRisco, Intencao, ParamsMotor, ResultadoMotor


MS = (0.20, 0.30, 0.50)
DEGRAUS = ((1, 1 / 3), (2, 2 / 3), (3, 1.0))
DIAS_ANO = 260.3
SEM_CAPACIDADE = {
    EstadoRisco.REDUCAO,
    EstadoRisco.DESCONHECIDA,
    EstadoRisco.REVOGADO,
}


@dataclass(frozen=True, slots=True)
class FonteCDI:
    """Taxa bruta anual em fração (por exemplo, 0,0959), com fonte auditável."""

    taxa_bruta_anual: float
    fonte: str
    referencia: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.taxa_bruta_anual) or self.taxa_bruta_anual < 0:
            raise ValueError("taxa CDI deve ser anual, finita e não negativa")
        if not self.fonte.strip() or not self.referencia.strip():
            raise ValueError("CDI exige fonte e referência declaradas")


@dataclass(frozen=True, slots=True)
class FonteCambio:
    """Identificação auditável da série USD/BRL injetada; nunca busca rede."""

    fonte: str
    referencia: str

    def __post_init__(self) -> None:
        if not self.fonte.strip() or not self.referencia.strip():
            raise ValueError("câmbio exige fonte e referência declaradas")


@dataclass(frozen=True, slots=True)
class PontoCDICambio:
    data: date
    fator_cdi_bruto: float
    usd_brl: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.fator_cdi_bruto) or self.fator_cdi_bruto <= 0:
            raise ValueError("fator diário do CDI deve ser finito e positivo")
        if not math.isfinite(self.usd_brl) or self.usd_brl <= 0:
            raise ValueError("USD/BRL deve ser finito e positivo")


@dataclass(frozen=True, slots=True)
class ComparacaoCDI:
    capital_inicial_usd: float
    multiplicador_cdi_bruto: float
    saldo_cdi_cambio_parado_usd: float
    saldo_cdi_cambio_historico_usd: float
    saldo_dolar_parado_usd: float
    aliquota_imposto: float
    fonte_cdi: FonteCDI
    fonte_cambio: FonteCambio


def comparar_cdi_cambios(
    capital_inicial_usd: float,
    pontos: Sequence[PontoCDICambio],
    fonte_cdi: FonteCDI,
    fonte_cambio: FonteCambio,
    *,
    aliquota_imposto: float = 0.15,
) -> ComparacaoCDI:
    """F2-6: publica as duas contas de câmbio do CDI, sem rede.

    O capital é convertido a BRL pelo primeiro USD/BRL e cada fator bruto do
    CDI é composto uma vez. O imposto incide somente sobre o ganho nominal do
    CDI. Na conta ``câmbio parado`` o saldo final volta a USD pela cotação
    inicial; na ``histórica``, pela última cotação observada. O dólar parado na
    corretora permanece no capital inicial e é publicado como terceira base.
    """
    if not math.isfinite(capital_inicial_usd) or capital_inicial_usd <= 0:
        raise ValueError("capital inicial deve ser finito e positivo")
    if not 0 <= aliquota_imposto < 1:
        raise ValueError("alíquota deve estar em [0, 1)")
    if not pontos:
        raise ValueError("comparação exige trajetória de CDI e USD/BRL")
    ordenados = sorted(pontos, key=lambda p: p.data)
    if len({p.data for p in ordenados}) != len(ordenados):
        raise ValueError("datas de CDI/câmbio não podem repetir")
    multiplicador = math.prod(p.fator_cdi_bruto for p in ordenados)
    cambio_inicial = ordenados[0].usd_brl
    principal_brl = capital_inicial_usd * cambio_inicial
    ganho_liquido_brl = principal_brl * (multiplicador - 1.0) * (1.0 - aliquota_imposto)
    saldo_brl = principal_brl + ganho_liquido_brl
    return ComparacaoCDI(
        capital_inicial_usd=capital_inicial_usd,
        multiplicador_cdi_bruto=multiplicador,
        saldo_cdi_cambio_parado_usd=saldo_brl / cambio_inicial,
        saldo_cdi_cambio_historico_usd=saldo_brl / ordenados[-1].usd_brl,
        saldo_dolar_parado_usd=capital_inicial_usd,
        aliquota_imposto=aliquota_imposto,
        fonte_cdi=fonte_cdi,
        fonte_cambio=fonte_cambio,
    )


@dataclass(frozen=True, slots=True)
class MetricasHorizonte:
    anos: int
    dias_observados: int
    revogou: bool
    resultado_usd: float
    resultado_sobre_capital: float
    queda_maxima_usd: float
    queda_maxima_fracao: float
    dias_sem_capacidade: int
    volatilidade_anual: float
    sharpe_equilibrio_cdi: float


@dataclass(frozen=True, slots=True)
class Partida:
    inicio: date
    m: float
    degrau: int
    fracao_degrau: float
    risco_aberto_alvo: float
    um_ano: MetricasHorizonte
    tres_anos: MetricasHorizonte
    resultado_acumulado_por_mes_usd: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FaixaResultadoIdade:
    cenario: str
    m: float
    degrau: int
    fracao_degrau: float
    risco_aberto_alvo: float
    idade_meses: int
    observacoes: int
    quantil_05_usd: float
    quantil_25_usd: float
    quantil_50_usd: float
    quantil_75_usd: float
    quantil_95_usd: float


@dataclass(frozen=True, slots=True)
class Agregado:
    m: float
    degrau: int
    fracao_degrau: float
    risco_aberto_alvo: float
    partidas: int
    chance_revogacao_1a: float
    chance_revogacao_3a: float
    resultado_mediano_1a_usd: float
    resultado_mediano_3a_usd: float
    resultado_mediano_1a_sobre_capital: float
    resultado_mediano_3a_sobre_capital: float
    queda_mediana_1a_usd: float
    queda_mediana_3a_usd: float
    queda_mediana_1a_fracao: float
    queda_mediana_3a_fracao: float
    dias_sem_capacidade_mediano_1a: float
    dias_sem_capacidade_mediano_3a: float
    volatilidade_mediana_3a: float
    sharpe_equilibrio_mediano_3a: float


@dataclass(frozen=True, slots=True)
class RelatorioViabilidade:
    fonte_cdi: FonteCDI
    risco_aberto_por_m: Mapping[float, float]
    partidas: tuple[Partida, ...]
    agregados: tuple[Agregado, ...]
    sharpe_plantado: float | None
    projeto_para_aqui: bool | None
    nota_risco_aberto: str


def _somar_anos(d: date, anos: int) -> date:
    try:
        return d.replace(year=d.year + anos)
    except ValueError:  # 29 de fevereiro
        return d.replace(year=d.year + anos, day=28)


def _somar_meses(d: date, meses: int) -> date:
    total = d.year * 12 + d.month - 1 + meses
    ano, mes0 = divmod(total, 12)
    mes = mes0 + 1
    return date(ano, mes, min(d.day, monthrange(ano, mes)[1]))


def _resultado_por_idade(
    resultado: ResultadoMotor, inicio: date, meses: int = 36
) -> tuple[float, ...]:
    return tuple(
        float(sum(
            pnl for d, pnl in zip(resultado.datas, resultado.pnl_usd)
            if inicio <= d < _somar_meses(inicio, idade)
        ))
        for idade in range(1, meses + 1)
    )


def _quantil_tipo_7(valores: Sequence[float], q: float) -> float:
    """Quantil linear tipo 7: h=(n-1)q, padrão do NumPy/R."""
    x = sorted(float(v) for v in valores)
    if not x:
        raise ValueError("quantil exige observações")
    h = (len(x) - 1) * q
    lo = math.floor(h)
    hi = math.ceil(h)
    return x[lo] + (h - lo) * (x[hi] - x[lo])


def faixas_resultado_por_idade(
    sem_vantagem: Sequence["RelatorioViabilidade"],
    com_vantagem: Sequence["RelatorioViabilidade"],
) -> tuple[FaixaResultadoIdade, ...]:
    """D4.5: distribuição do resultado acumulado por idade da conta.

    Uma observação é uma partida mensal do sistema completo. Para idade
    ``j`` (1..36), o resultado é a soma do P&L em USD no intervalo
    ``[inicio_da_partida, inicio_da_partida + j meses)``; portanto inclui o
    reinvestimento que o próprio motor composto já aplicou, sem anualização.
    As partidas sobrepostas permanecem observações descritivas. Agrupa-se por
    cenário (``sem_vantagem``/``com_vantagem``), M e degrau, juntando todos os
    mundos entregues. Publicam-se 5/25/50/75/95% pelo quantil linear tipo 7,
    ``h=(n-1)q``. Esta faixa é diagnóstico mensal e não decide o portão.
    """
    if not sem_vantagem or not com_vantagem:
        raise ValueError("faixa por idade exige mundos com e sem vantagem")
    saida: list[FaixaResultadoIdade] = []
    for cenario, relatorios in (
        ("sem_vantagem", sem_vantagem),
        ("com_vantagem", com_vantagem),
    ):
        partidas = [p for r in relatorios for p in r.partidas]
        for m in MS:
            for degrau, fracao in DEGRAUS:
                grupo = [p for p in partidas if p.m == m and p.degrau == degrau]
                if not grupo:
                    continue
                if any(len(p.resultado_acumulado_por_mes_usd) != 36 for p in grupo):
                    raise ValueError("partida sem as 36 idades mensais de D4.5")
                for idade in range(1, 37):
                    valores = [
                        p.resultado_acumulado_por_mes_usd[idade - 1]
                        for p in grupo
                    ]
                    qs = tuple(
                        _quantil_tipo_7(valores, q)
                        for q in (0.05, 0.25, 0.50, 0.75, 0.95)
                    )
                    saida.append(FaixaResultadoIdade(
                        cenario, m, degrau, fracao,
                        grupo[0].risco_aberto_alvo, idade, len(valores), *qs,
                    ))
    return tuple(saida)


def _primeiros_pregoes_mensais(dias: Sequence[date], inicio: date, fim: date) -> tuple[date, ...]:
    primeiros: dict[tuple[int, int], date] = {}
    for d in sorted(set(dias)):
        if inicio <= d <= fim:
            primeiros.setdefault((d.year, d.month), d)
    return tuple(primeiros.values())


class _DiasIntencoes:
    """Resolve o dia de servidor sem confundir a data UTC do sinal."""

    def __init__(self, mundo: Mundo):
        self._por_simbolo: dict[str, tuple[list[datetime], list[object]]] = {}
        for simbolo, por_dia in mundo.sessoes.items():
            sessoes = sorted(por_dia.values(), key=lambda s: s.abertura_utc)
            self._por_simbolo[simbolo] = ([s.abertura_utc for s in sessoes], sessoes)

    def __call__(self, i: Intencao) -> date:
        aberturas, sessoes = self._por_simbolo[i.simbolo]
        k = bisect_right(aberturas, i.barra_sinal_fim) - 1
        if k < 0:
            raise ValueError(f"intenção {i.intencao_id} fora das sessões")
        s = sessoes[k]
        if not (s.abertura_utc <= i.barra_sinal_fim < s.virada_utc):
            raise ValueError(f"intenção {i.intencao_id} fora da sessão do servidor")
        return s.data_servidor


class _DiasEstados:
    """Traduz instantes globais do motor para datas do servidor."""

    def __init__(self, mundo: Mundo):
        intervalos: dict[date, tuple[datetime, datetime]] = {}
        for por_dia in mundo.sessoes.values():
            for d, s in por_dia.items():
                if d not in intervalos:
                    intervalos[d] = (s.abertura_utc, s.virada_utc)
                else:
                    a, v = intervalos[d]
                    intervalos[d] = (min(a, s.abertura_utc), max(v, s.virada_utc))
        ordenados = sorted((a, v, d) for d, (a, v) in intervalos.items())
        self._aberturas = [x[0] for x in ordenados]
        self._intervalos = ordenados

    def __call__(self, instante: datetime) -> date:
        k = bisect_right(self._aberturas, instante) - 1
        if k >= 0:
            a, v, d = self._intervalos[k]
            if a <= instante < v:
                return d
        # Estados só deveriam nascer em barras conhecidas. O fallback explícito
        # preserva utilidade para carregadores de teste sem calendário completo.
        return instante.date()


def _metricas(
    resultado: ResultadoMotor,
    inicio: date,
    limite: date,
    fonte_cdi: FonteCDI,
    capital: float,
    dia_servidor=lambda t: t.date(),
) -> MetricasHorizonte:
    pares = [(d, p) for d, p in zip(resultado.datas, resultado.pnl_usd) if inicio <= d < limite]
    patrimonio = capital
    pico = capital
    queda_usd = 0.0
    queda_fracao = 0.0
    retornos = []
    for _, pnl in pares:
        anterior = patrimonio
        patrimonio += pnl
        retornos.append(pnl / anterior if anterior > 0 else 0.0)
        pico = max(pico, patrimonio)
        queda = pico - patrimonio
        queda_usd = max(queda_usd, queda)
        queda_fracao = max(queda_fracao, queda / pico if pico > 0 else 1.0)

    estados = [(t, e) for t, e in resultado.estados if inicio <= dia_servidor(t) < limite]
    dias_sem = {dia_servidor(t) for t, e in estados if e in SEM_CAPACIDADE}
    revogou = any(e is EstadoRisco.REVOGADO for _, e in estados)
    vol = stdev(retornos) * math.sqrt(DIAS_ANO) if len(retornos) >= 2 else 0.0
    equilibrio = fonte_cdi.taxa_bruta_anual / vol if vol > 0 else math.inf
    anos = max(1, round((limite - inicio).days / 365.25))
    return MetricasHorizonte(
        anos, len(pares), revogou, patrimonio - capital, (patrimonio - capital) / capital,
        queda_usd, queda_fracao, len(dias_sem), vol, equilibrio,
    )


def _agregar(partidas: Sequence[Partida]) -> tuple[Agregado, ...]:
    saida = []
    for m in MS:
        for degrau, fracao in DEGRAUS:
            grupo = [p for p in partidas if p.m == m and p.degrau == degrau]
            if not grupo:
                continue
            n = len(grupo)
            eq = [p.tres_anos.sharpe_equilibrio_cdi for p in grupo]
            saida.append(Agregado(
                m, degrau, fracao, grupo[0].risco_aberto_alvo, n,
                sum(p.um_ano.revogou for p in grupo) / n,
                sum(p.tres_anos.revogou for p in grupo) / n,
                median(p.um_ano.resultado_usd for p in grupo),
                median(p.tres_anos.resultado_usd for p in grupo),
                median(p.um_ano.resultado_sobre_capital for p in grupo),
                median(p.tres_anos.resultado_sobre_capital for p in grupo),
                median(p.um_ano.queda_maxima_usd for p in grupo),
                median(p.tres_anos.queda_maxima_usd for p in grupo),
                median(p.um_ano.queda_maxima_fracao for p in grupo),
                median(p.tres_anos.queda_maxima_fracao for p in grupo),
                median(p.um_ano.dias_sem_capacidade for p in grupo),
                median(p.tres_anos.dias_sem_capacidade for p in grupo),
                median(p.tres_anos.volatilidade_anual for p in grupo),
                median(eq),
            ))
    return tuple(saida)


def avaliar_viabilidade(
    mundo: Mundo,
    intencoes: Sequence[Intencao],
    params_base: ParamsMotor,
    risco_aberto_por_m: Mapping[float, float],
    fonte_cdi: FonteCDI,
    *,
    janela_inicio: date | None = None,
    janela_fim: date | None = None,
    sharpe_plantado: float | None = None,
) -> RelatorioViabilidade:
    """Simula partidas mensais completas de três anos, por M e degrau.

    ``risco_aberto_por_m`` contém frações do orçamento M×H, não frações do
    patrimônio. O degrau multiplica essa fração por 1/3, 2/3 ou 1.
    ``sharpe_plantado=1`` ativa a regra de parada de F2-10; nos demais mundos o
    campo correspondente fica ``None``.
    """
    if params_base.capital_constante:
        raise ValueError("F2-10 exige ParamsMotor(capital_constante=False)")
    if set(risco_aberto_por_m) != set(MS):
        raise ValueError("risco_aberto_por_m deve conter exatamente M=0,20/0,30/0,50")
    if any(not math.isfinite(v) or not 0 < v <= 1 for v in risco_aberto_por_m.values()):
        raise ValueError("alvos de risco aberto devem estar em (0, 1]")
    if not mundo.dias:
        raise ValueError("mundo sem pregões")

    inicio = janela_inicio or min(mundo.dias)
    fim = janela_fim or max(mundo.dias)
    if inicio > fim:
        raise ValueError("janela invertida")
    inicios = tuple(
        d for d in _primeiros_pregoes_mensais(mundo.dias, inicio, fim)
        if _somar_anos(d, 3) <= fim
    )
    if not inicios:
        raise ValueError("janela não contém partida mensal com três anos completos")

    dia_intencao = _DiasIntencoes(mundo)
    dia_estado = _DiasEstados(mundo)
    intencoes_dia = [(i, dia_intencao(i)) for i in intencoes]
    simbolos = sorted(mundo.m15)
    partidas: list[Partida] = []
    for partida in inicios:
        limite_1 = _somar_anos(partida, 1)
        limite_3 = _somar_anos(partida, 3)
        its = [i for i, d in intencoes_dia if partida <= d < limite_3]
        carregador = {
            s: tuple(b for b in mundo.barras_motor(s) if b.data_servidor < limite_3)
            for s in simbolos
        }
        for m in MS:
            for degrau, fracao in DEGRAUS:
                alvo = risco_aberto_por_m[m] * fracao
                params = replace(params_base, limiar_m=m, risco_aberto_alvo=alvo, capital_constante=False)
                resultado = simular(its, carregador, mundo.specs, mundo.calendario, params)
                partidas.append(Partida(
                    partida, m, degrau, fracao, alvo,
                    _metricas(resultado, partida, limite_1, fonte_cdi, params.capital_usd, dia_estado),
                    _metricas(resultado, partida, limite_3, fonte_cdi, params.capital_usd, dia_estado),
                    _resultado_por_idade(resultado, partida),
                ))

    agregados = _agregar(partidas)
    parar = None
    if sharpe_plantado is not None and math.isclose(sharpe_plantado, 1.0, abs_tol=1e-12):
        inteiros = [a for a in agregados if a.degrau == 3]
        parar = bool(inteiros) and all(a.resultado_mediano_3a_usd <= 0 for a in inteiros)
    return RelatorioViabilidade(
        fonte_cdi, dict(risco_aberto_por_m), tuple(partidas), agregados,
        sharpe_plantado, parar,
        "Alvo por M é entrada da calibração sem vantagem: maior valor com CP95% da revogação em 3 anos <= 10%.",
    )


__all__ = [
    "Agregado", "ComparacaoCDI", "FaixaResultadoIdade", "FonteCDI", "FonteCambio",
    "MetricasHorizonte", "PontoCDICambio", "comparar_cdi_cambios",
    "Partida", "RelatorioViabilidade", "avaliar_viabilidade",
    "faixas_resultado_por_idade",
]
