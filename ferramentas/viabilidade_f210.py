"""F2-10: viabilidade do sistema completo em mundos sintéticos.

Este módulo não escolhe o risco aberto alvo. A seção 4 manda calibrá-lo como o
maior valor cuja revogação em três anos, em mundos sem vantagem, tenha limite
superior CP95% de no máximo 10%. Por isso ``risco_aberto_por_m`` é uma entrada
obrigatória, já produzida por essa calibração.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from calendar import monthrange
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
import json
import math
from pathlib import Path
from statistics import median, stdev
import sys
import time
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

# Desenho oficial congelado antes da primeira execucao F2-10. Os 29 mundos
# independentes por cenario igualam a populacao da calibracao do risco aberto;
# as partidas mensais sobrepostas sao observacoes descritivas, nao replicas.
VARIANTE_OFICIAL = "V3"
RISCO_OFICIAL = 0.02
N_ATIVOS = 81
N_DIAS = 1_240
N_MUNDOS_POR_CENARIO = 29
CENARIOS = ("sem_vantagem", "com_vantagem_sharpe_1")
N_TAREFAS = len(CENARIOS) * N_MUNDOS_POR_CENARIO
REAMOSTRAGENS = 499
PESSIMISTA = True
SEMENTE_NULA_BASE = 340_000
SEMENTE_SELECAO_NULA_BASE = 440_000
SEMENTE_VANTAGEM_BASE = 540_000
PILOTO_8_ATIVOS_SEGUNDOS = 661.4
FORCAS_PATH = Path(__file__).resolve().parents[1] / "medidas" / "f2_12_forcas_vantagem.json"
RISCO_ABERTO_PATH = Path(__file__).resolve().parents[1] / "medidas" / "f2_10_risco_aberto.json"
SAIDA_PADRAO = Path(__file__).resolve().parents[1] / "medidas" / "f2_10_viabilidade.json"


def numero_tarefas(risco_aberto_por_m: Mapping[float, float | None]) -> int:
    """Não dispara mundos quando a grade não deu alvo a nenhum M."""
    return N_TAREFAS if any(v is not None for v in risco_aberto_por_m.values()) else 0


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


FONTE_CDI_OFICIAL = FonteCDI(
    0.0959,
    "serie do Banco Central registrada no Plano v29",
    "15/12/2014 a 09/10/2025; taxa bruta anual medida",
)


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
    risco_aberto_por_m: Mapping[float, float | None]
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
    risco_aberto_por_m: Mapping[float, float | None],
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
    if any(v is not None and (not math.isfinite(v) or not 0 < v <= 1)
           for v in risco_aberto_por_m.values()):
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
            if risco_aberto_por_m[m] is None:
                continue
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
        parar = all(a.resultado_mediano_3a_usd <= 0 for a in inteiros)
    return RelatorioViabilidade(
        fonte_cdi, dict(risco_aberto_por_m), tuple(partidas), agregados,
        sharpe_plantado, parar,
        "Alvo por M é entrada da calibração sem vantagem: maior valor com CP95% da revogação em 3 anos <= 10%.",
    )


def _ler_risco_aberto(path: Path) -> tuple[dict[float, float | None], dict[str, dict]]:
    """Carrega alvos e registra o primeiro ponto reprovado de cada M sem alvo."""
    if not path.is_file():
        raise FileNotFoundError(f"resultado do risco aberto ausente: {path}")
    bruto = json.loads(path.read_text(encoding="utf-8"))
    if bruto.get("status") != "completo":
        raise RuntimeError("calibracao do risco aberto ausente ou em_andamento")
    try:
        declarados = bruto["alvo_por_m"]
        if set(declarados) != {f"{m:.2f}" for m in MS}:
            raise KeyError("chaves")
        alvos = {
            m: None if declarados[f"{m:.2f}"] is None else float(declarados[f"{m:.2f}"])
            for m in MS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("alvo_por_m incompleto no resultado do risco aberto") from exc
    if any(x is not None and (not math.isfinite(x) or not 0 < x <= 1) for x in alvos.values()):
        raise ValueError("alvos do risco aberto devem estar em (0, 1] ou ser null")

    pontos = bruto.get("pontos")
    situacao: dict[str, dict] = {}
    for m in MS:
        chave = f"{m:.2f}"
        if alvos[m] is not None:
            situacao[chave] = {"status": "alvo_viavel", "alvo": alvos[m]}
            continue
        candidatos = [
            p for p in pontos or ()
            if isinstance(p, dict) and p.get("m") == m and p.get("aprovado") is False
        ]
        if not candidatos:
            raise ValueError(f"M={chave} sem alvo viavel e sem ponto reprovado")
        try:
            ponto = min(candidatos, key=lambda p: float(p["alvo"]))
            reprovado = {
                "alvo": float(ponto["alvo"]),
                "k_mundos": int(ponto["k_mundos"]),
                "n_mundos": int(ponto["n_mundos"]),
                "limite_cp95": float(ponto["limite_cp95"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"ponto reprovado incompleto para M={chave}") from exc
        situacao[chave] = {"status": "sem_alvo_viavel", "ponto_reprovado": reprovado}
    return alvos, situacao


def carregar_risco_aberto(path: Path = RISCO_ABERTO_PATH) -> dict[float, float | None]:
    """Le o resultado concluido da calibracao; ausencia ou parcial falha fechado."""
    return _ler_risco_aberto(path)[0]


def _forca_momento(sharpe: float) -> float:
    bruto = json.loads(FORCAS_PATH.read_text(encoding="utf-8"))
    try:
        valor = float(bruto["forcas"]["momento"][f"{sharpe:.1f}"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"forca congelada de momento Sharpe {sharpe:g} ausente") from exc
    if not math.isfinite(valor) or valor <= 0:
        raise ValueError("forca plantada deve ser positiva e finita")
    return valor


def projecao_custo() -> dict[str, float | str]:
    """Extrapolacao linear do pior caso medido: tarefa nula condicionada."""
    segundos_81 = PILOTO_8_ATIVOS_SEGUNDOS * N_ATIVOS / 8
    return {
        "piloto_8_ativos_segundos": PILOTO_8_ATIVOS_SEGUNDOS,
        "piloto": "cenario nulo; 1240 dias; preparo de selecao e execucao; 90 simulacoes",
        "estimativa_linear_81_ativos_minutos_notebook": round(segundos_81 / 60, 2),
        "estimativa_gitHub_minutos_com_fator_2x_medido_no_f2_12": round(segundos_81 / 120, 2),
    }


def _dias_ordinais_lote(mundo: Mundo, lote):
    """Data de servidor por linha compacta, sem materializar Intencao."""
    import numpy as np

    sessoes = sorted(mundo.sessoes[lote.simbolo].values(), key=lambda s: s.abertura_utc)
    para_ns = lambda x: int(x.timestamp()) * 10**9 + x.microsecond * 1000
    aberturas = np.array([para_ns(s.abertura_utc) for s in sessoes], dtype=np.int64)
    viradas = np.array([para_ns(s.virada_utc) for s in sessoes], dtype=np.int64)
    dias = np.array([s.data_servidor.toordinal() for s in sessoes], dtype=np.int32)
    pos = np.searchsorted(aberturas, lote.sinal_fim_ns, side="right") - 1
    if np.any(pos < 0) or np.any(lote.sinal_fim_ns >= viradas[pos]):
        raise ValueError(f"intencao compacta fora das sessoes de {lote.simbolo}")
    return np.ascontiguousarray(dias[pos])


@dataclass(slots=True)
class _PreparadoOficial:
    mundo: Mundo
    lotes: Mapping[str, object]
    dias_ordinais: Mapping[str, object]
    params_base: ParamsMotor
    cenario: str
    semente_execucao: int
    semente_selecao: int


def _preparar_cenario(
    cenario: str,
    numero: int,
    processos: int = 1,
    *,
    n_ativos: int = N_ATIVOS,
    n_dias: int = N_DIAS,
) -> _PreparadoOficial:
    """Prepara V3/2% compacto; no nulo condiciona a sistema selecionado.

    O nulo repete o desenho independente da calibracao do risco aberto: agenda
    aprendida em momento Sharpe 2 e execucao sem vantagem. No Sharpe 1,
    aprendizagem e execucao ocorrem causalmente no proprio mundo plantado.
    """
    from titan import cadeia_garimpo as cg
    from titan.aprendizagem import agenda, aplicar_compacto
    from titan.sintetico import Vantagem, gerar_mundo

    if cenario not in CENARIOS or not 0 <= numero < N_MUNDOS_POR_CENARIO:
        raise ValueError("coordenada de mundo F2-10 invalida")
    if processos < 1:
        raise ValueError("processos deve ser positivo")
    if cenario == "sem_vantagem":
        semente_execucao = SEMENTE_NULA_BASE + numero
        semente_selecao = SEMENTE_SELECAO_NULA_BASE + numero
        selecao = gerar_mundo(
            semente_selecao, n_dias, n_ativos,
            Vantagem("momento", _forca_momento(2.0), 0.5),
        )
        brutas_sel, por_risco = cg.preparar(selecao, (RISCO_OFICIAL,), processos=processos)
        periodos = agenda(
            VARIANTE_OFICIAL, selecao.dias, por_risco[RISCO_OFICIAL],
            reamostragens=REAMOSTRAGENS, semente=semente_selecao,
        )
        simbolos = set(selecao.m15)
        mundo = gerar_mundo(semente_execucao, n_dias, n_ativos, Vantagem())
        if simbolos != set(mundo.m15):
            raise ValueError("simbolos dos mundos de selecao e execucao divergem")
        brutas, _ = cg.preparar(mundo, (), processos=processos)
        if set(brutas_sel) != set(brutas):
            raise ValueError("lotes dos mundos de selecao e execucao divergem")
    else:
        semente_execucao = SEMENTE_VANTAGEM_BASE + numero
        semente_selecao = semente_execucao
        mundo = gerar_mundo(
            semente_execucao, n_dias, n_ativos,
            Vantagem("momento", _forca_momento(1.0), 0.5),
        )
        brutas, por_risco = cg.preparar(mundo, (RISCO_OFICIAL,), processos=processos)
        periodos = agenda(
            VARIANTE_OFICIAL, mundo.dias, por_risco[RISCO_OFICIAL],
            reamostragens=REAMOSTRAGENS, semente=semente_execucao,
        )
    lotes = aplicar_compacto(brutas, periodos)
    if sum(map(len, lotes.values())) == 0:
        raise RuntimeError(f"V3 nao emitiu intencoes no cenario {cenario}")
    return _PreparadoOficial(
        mundo, lotes, {s: _dias_ordinais_lote(mundo, x) for s, x in lotes.items()},
        replace(cg.params_motor(RISCO_OFICIAL, PESSIMISTA), capital_constante=False),
        cenario, semente_execucao, semente_selecao,
    )


def _avaliar_compacto(
    prep: _PreparadoOficial,
    risco_aberto_por_m: Mapping[float, float | None],
    fonte_cdi: FonteCDI = FONTE_CDI_OFICIAL,
) -> RelatorioViabilidade:
    import numpy as np
    from titan import cadeia_garimpo as cg
    from titan.intencoes_compactas import ColecaoIntencoesCompactas

    if set(risco_aberto_por_m) != set(MS):
        raise ValueError("risco_aberto_por_m deve conter exatamente os tres M")
    if not any(v is not None for v in risco_aberto_por_m.values()):
        raise ValueError("nenhum M tem alvo viavel; nao ha mundo para simular")
    mundo = prep.mundo
    janela = cg.janelas_padrao(mundo.dias)
    fim = max(mundo.dias)
    inicios = tuple(
        d for d in _primeiros_pregoes_mensais(mundo.dias, janela.janela_inicio, fim)
        if _somar_anos(d, 3) <= fim
    )
    if not inicios:
        raise ValueError("janela nao contem partida mensal com tres anos completos")
    dia_estado = _DiasEstados(mundo)
    partidas: list[Partida] = []
    for inicio in inicios:
        limite_1, limite_3 = _somar_anos(inicio, 1), _somar_anos(inicio, 3)
        selecionados = []
        for simbolo in sorted(prep.lotes):
            dias = prep.dias_ordinais[simbolo]
            linhas = np.flatnonzero(
                (dias >= inicio.toordinal()) & (dias < limite_3.toordinal())
            ).astype(np.int64)
            if len(linhas):
                selecionados.append(prep.lotes[simbolo].selecionar(np.ascontiguousarray(linhas)))
        its = ColecaoIntencoesCompactas(selecionados)
        carregador = {
            lote.simbolo: tuple(
                b for b in mundo.barras_motor(lote.simbolo) if b.data_servidor < limite_3
            ) for lote in selecionados
        }
        for m in MS:
            if risco_aberto_por_m[m] is None:
                continue
            for degrau, fracao in DEGRAUS:
                alvo = risco_aberto_por_m[m] * fracao
                params = replace(
                    prep.params_base, limiar_m=m, risco_aberto_alvo=alvo,
                    capital_constante=False,
                )
                resultado = simular(its, carregador, mundo.specs, mundo.calendario, params)
                partidas.append(Partida(
                    inicio, m, degrau, fracao, alvo,
                    _metricas(resultado, inicio, limite_1, fonte_cdi, params.capital_usd, dia_estado),
                    _metricas(resultado, inicio, limite_3, fonte_cdi, params.capital_usd, dia_estado),
                    _resultado_por_idade(resultado, inicio),
                ))
    sharpe = 1.0 if prep.cenario == "com_vantagem_sharpe_1" else None
    agregados = _agregar(partidas)
    parar = None
    if sharpe == 1.0:
        inteiros = [a for a in agregados if a.degrau == 3]
        parar = all(a.resultado_mediano_3a_usd <= 0 for a in inteiros)
    return RelatorioViabilidade(
        fonte_cdi, dict(risco_aberto_por_m), tuple(partidas), agregados, sharpe, parar,
        "Alvo por M lido da calibracao concluida sem vantagem; execucao pessimista.",
    )


def _jsonificavel(valor):
    if isinstance(valor, date):
        return valor.isoformat()
    if isinstance(valor, float) and not math.isfinite(valor):
        return "Infinity" if valor > 0 else "-Infinity"
    if isinstance(valor, Mapping):
        return {str(k): _jsonificavel(v) for k, v in valor.items()}
    if isinstance(valor, (tuple, list)):
        return [_jsonificavel(x) for x in valor]
    return valor


def _relatorio_para_dict(r: RelatorioViabilidade) -> dict:
    return _jsonificavel(asdict(r))


def _relatorio_de_dict(d: Mapping) -> RelatorioViabilidade:
    def metrica(x):
        x = dict(x)
        if x["sharpe_equilibrio_cdi"] in ("Infinity", "-Infinity"):
            x["sharpe_equilibrio_cdi"] = float(x["sharpe_equilibrio_cdi"].replace("Infinity", "inf"))
        return MetricasHorizonte(**x)
    partidas = tuple(Partida(
        date.fromisoformat(x["inicio"]), float(x["m"]), int(x["degrau"]),
        float(x["fracao_degrau"]), float(x["risco_aberto_alvo"]),
        metrica(x["um_ano"]), metrica(x["tres_anos"]),
        tuple(map(float, x["resultado_acumulado_por_mes_usd"])),
    ) for x in d["partidas"])
    return RelatorioViabilidade(
        FonteCDI(**d["fonte_cdi"]),
        {float(k): None if v is None else float(v) for k, v in d["risco_aberto_por_m"].items()},
        partidas, _agregar(partidas), d.get("sharpe_plantado"),
        d.get("projeto_para_aqui"), d["nota_risco_aberto"],
    )


def _consolidar(
    resultados: Sequence[tuple[str, int, RelatorioViabilidade]],
    risco_aberto_por_m: Mapping[float, float | None],
    situacao_por_m: Mapping[str, Mapping] | None = None,
) -> dict:
    ha_alvo = numero_tarefas(risco_aberto_por_m) > 0
    por_cenario = {c: [r for c0, _, r in resultados if c0 == c] for c in CENARIOS}
    esperado = N_MUNDOS_POR_CENARIO if ha_alvo else 0
    if any(len(por_cenario[c]) != esperado for c in CENARIOS):
        raise ValueError("consolidacao exige todos os mundos dos dois cenarios")
    agregados = {}
    for c in CENARIOS:
        partidas = tuple(p for r in por_cenario[c] for p in r.partidas)
        agregados[c] = _jsonificavel([asdict(a) for a in _agregar(partidas)])
    vantagem_ag = _agregar(tuple(
        p for r in por_cenario["com_vantagem_sharpe_1"] for p in r.partidas
    ))
    inteiros = [a for a in vantagem_ag if a.degrau == 3]
    # M sem alvo conta como não positivo; logo basta algum M simulado positivo
    # para continuar, e a ausência total de alvos para imediatamente.
    parar = all(a.resultado_mediano_3a_usd <= 0 for a in inteiros)
    faixas = faixas_resultado_por_idade(
        por_cenario["sem_vantagem"], por_cenario["com_vantagem_sharpe_1"],
    ) if ha_alvo else ()
    if situacao_por_m is None:
        situacao_por_m = {
            f"{m:.2f}": {"status": "alvo_viavel", "alvo": risco_aberto_por_m[m]}
            for m in MS if risco_aberto_por_m[m] is not None
        }
    return _jsonificavel({
        "status": "completo",
        "configuracao": {
            "variante": VARIANTE_OFICIAL, "risco_por_operacao": RISCO_OFICIAL,
            "n_ativos": N_ATIVOS, "n_dias": N_DIAS, "pessimista": PESSIMISTA,
            "reamostragens": REAMOSTRAGENS,
        },
        "desenho": {
            "mundos_por_cenario": N_MUNDOS_POR_CENARIO,
            "mundos_executados_por_cenario": esperado,
            "sem_vantagem": "agenda aprendida em momento Sharpe 2 independente; execucao nula condicional a sistema selecionado",
            "com_vantagem": "momento Sharpe 1; aprendizagem causal e execucao no mesmo mundo",
            "forcas": {
                "selecao_nula_momento_sharpe_2": _forca_momento(2.0),
                "vantagem_momento_sharpe_1": _forca_momento(1.0),
                "fracao_ativos": 0.5,
            },
            "sementes": {
                "execucao_nula": [SEMENTE_NULA_BASE, SEMENTE_NULA_BASE + N_MUNDOS_POR_CENARIO - 1],
                "selecao_nula_sharpe_2": [SEMENTE_SELECAO_NULA_BASE, SEMENTE_SELECAO_NULA_BASE + N_MUNDOS_POR_CENARIO - 1],
                "vantagem_sharpe_1": [SEMENTE_VANTAGEM_BASE, SEMENTE_VANTAGEM_BASE + N_MUNDOS_POR_CENARIO - 1],
            },
            "unidade": "mundo independente; partidas mensais sobrepostas sao descritivas",
        },
        "fonte_cdi": asdict(FONTE_CDI_OFICIAL),
        "risco_aberto_por_m": {f"{m:.2f}": risco_aberto_por_m[m] for m in MS},
        "situacao_por_m": situacao_por_m,
        "agregados_por_cenario": agregados,
        "faixas_resultado_por_idade": [asdict(x) for x in faixas],
        "regra_parada": {
            "definicao": "para se nenhum M no degrau inteiro tem resultado mediano positivo em 3 anos sob Sharpe 1; M sem alvo viavel conta como nao positivo",
            "projeto_para_aqui": parar,
        },
        "limitacao_poder_f2_12": "V3 a 2% teve poder medio oficial de 5,0% nos nove cenarios plantados.",
        "projecao_custo": projecao_custo(),
    })


def executar_sequencial(
    avaliar_mundo, risco_aberto_por_m: Mapping[float, float | None],
) -> dict:
    resultados = []
    if numero_tarefas(risco_aberto_por_m):
        for cenario in CENARIOS:
            for numero in range(N_MUNDOS_POR_CENARIO):
                resultados.append((cenario, numero, avaliar_mundo(cenario, numero, risco_aberto_por_m)))
    return _consolidar(resultados, risco_aberto_por_m)


def _coordenadas_tarefa(tarefa: int) -> tuple[str, int]:
    if not 0 <= tarefa < N_TAREFAS:
        raise ValueError(f"tarefa deve estar entre 0 e {N_TAREFAS - 1}")
    return CENARIOS[tarefa // N_MUNDOS_POR_CENARIO], tarefa % N_MUNDOS_POR_CENARIO


def executar_tarefa(
    tarefa: int, proc_interno: int, saida: Path, *,
    risco_path: Path = RISCO_ABERTO_PATH,
    n_ativos: int = N_ATIVOS, n_dias: int = N_DIAS,
) -> dict:
    cenario, numero = _coordenadas_tarefa(tarefa)
    riscos, _ = _ler_risco_aberto(risco_path)
    if not numero_tarefas(riscos):
        raise RuntimeError("nenhum M tem alvo viavel; esta tarefa nao deve ser executada")
    t0 = time.time()
    prep = _preparar_cenario(cenario, numero, proc_interno, n_ativos=n_ativos, n_dias=n_dias)
    relatorio = _avaliar_compacto(prep, riscos)
    linha = {
        "tarefa": tarefa, "cenario": cenario, "numero": numero,
        "semente_execucao": prep.semente_execucao, "semente_selecao": prep.semente_selecao,
        "variante": VARIANTE_OFICIAL, "risco": RISCO_OFICIAL,
        "n_ativos": n_ativos, "n_dias": n_dias,
        "risco_aberto_por_m": {f"{m:.2f}": riscos[m] for m in MS},
        "relatorio": _relatorio_para_dict(relatorio),
        "segundos": round(time.time() - t0, 1),
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    tmp = saida.with_suffix(saida.suffix + ".tmp")
    tmp.write_text(json.dumps(linha, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(saida)
    return linha


def agregar_tarefas(
    pasta: Path, saida: Path = SAIDA_PADRAO, *, risco_path: Path = RISCO_ABERTO_PATH,
) -> dict:
    riscos, situacao = _ler_risco_aberto(risco_path)
    if not numero_tarefas(riscos):
        artefatos = [p for p in pasta.rglob("*.jsonl") if p.read_text(encoding="utf-8").strip()]
        if artefatos:
            raise ValueError("nenhum M tem alvo viavel, mas ha artefatos de tarefas")
        doc = _consolidar((), riscos, situacao)
        saida.parent.mkdir(parents=True, exist_ok=True)
        tmp = saida.with_suffix(saida.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(saida)
        return doc
    por_tarefa = {}
    for path in sorted(pasta.rglob("*.jsonl")):
        for bruto in path.read_text(encoding="utf-8").splitlines():
            if not bruto.strip():
                continue
            d = json.loads(bruto)
            tarefa = int(d["tarefa"])
            if tarefa in por_tarefa:
                raise ValueError(f"tarefa repetida: {tarefa}")
            cenario, numero = _coordenadas_tarefa(tarefa)
            s_exec = SEMENTE_NULA_BASE + numero if cenario == "sem_vantagem" else SEMENTE_VANTAGEM_BASE + numero
            s_sel = SEMENTE_SELECAO_NULA_BASE + numero if cenario == "sem_vantagem" else s_exec
            esperado = (
                cenario, numero, s_exec, s_sel, VARIANTE_OFICIAL, RISCO_OFICIAL,
                N_ATIVOS, N_DIAS, {f"{m:.2f}": riscos[m] for m in MS},
            )
            obtido = (
                d["cenario"], d["numero"], d["semente_execucao"], d["semente_selecao"],
                d["variante"], d["risco"], d["n_ativos"], d["n_dias"], d["risco_aberto_por_m"],
            )
            if obtido != esperado:
                raise ValueError(f"cabecalho divergente na tarefa {tarefa}")
            por_tarefa[tarefa] = d
    faltantes = sorted(set(range(N_TAREFAS)) - set(por_tarefa))
    if faltantes:
        raise ValueError(f"faltam {len(faltantes)} tarefas; primeira: {faltantes[0]}")
    resultados = []
    for tarefa in range(N_TAREFAS):
        cenario, numero = _coordenadas_tarefa(tarefa)
        relatorio = _relatorio_de_dict(por_tarefa[tarefa]["relatorio"])
        sharpe_esperado = 1.0 if cenario == "com_vantagem_sharpe_1" else None
        if relatorio.risco_aberto_por_m != dict(riscos) or relatorio.sharpe_plantado != sharpe_esperado:
            raise ValueError(f"relatorio divergente na tarefa {tarefa}")
        resultados.append((cenario, numero, relatorio))
    doc = _consolidar(resultados, riscos, situacao)
    saida.parent.mkdir(parents=True, exist_ok=True)
    tmp = saida.with_suffix(saida.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(saida)
    return doc


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--listar", action="store_true")
    g.add_argument("--tarefa", type=int)
    g.add_argument("--agregar", type=Path)
    p.add_argument("--proc-interno", type=int, default=1)
    p.add_argument("saida", nargs="?", type=Path)
    a = p.parse_args()
    if a.listar:
        if a.saida is not None:
            p.error("--listar nao aceita saida")
        riscos = carregar_risco_aberto(RISCO_ABERTO_PATH)
        print(numero_tarefas(riscos))
    elif a.tarefa is not None:
        if a.saida is None:
            p.error("--tarefa exige arquivo JSONL")
        print(json.dumps(executar_tarefa(a.tarefa, a.proc_interno, a.saida), ensure_ascii=False))
    else:
        if a.saida is not None:
            p.error("--agregar grava somente na saida oficial")
        doc = agregar_tarefas(a.agregar)
        print(json.dumps({"saida": str(SAIDA_PADRAO), "projeto_para_aqui": doc["regra_parada"]["projeto_para_aqui"]}))


__all__ = [
    "Agregado", "ComparacaoCDI", "FaixaResultadoIdade", "FonteCDI", "FonteCambio",
    "MetricasHorizonte", "PontoCDICambio", "comparar_cdi_cambios",
    "Partida", "RelatorioViabilidade", "avaliar_viabilidade",
    "faixas_resultado_por_idade", "carregar_risco_aberto",
    "executar_tarefa", "agregar_tarefas", "executar_sequencial",
    "numero_tarefas", "projecao_custo",
]


if __name__ == "__main__":
    main()
