"""Executor da calibração F2-9 do ato inteiro em mundos sem vantagem.

O catálogo de sessões e contratos entra por injeção. Este módulo nunca abre
preços nem materializa o catálogo real; a porta do ensaio cego continua sendo
o único lugar em que os metadados persistidos são ligados ao ato.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.certificacao import limite_superior, n_minimo
from titan.mundo_f29 import FamiliaNula, MundoF29, gerar_mundo
from titan.tipos import Sessao
from titan.tipos_f2 import (
    Campanha, EntradaProva, Intencao, ParamsPortao, ResultadoMotor,
    VereditoPortao,
)
from titan.motor import EspecificacaoAtivo


TETO_FALSO_POSITIVO = 0.025
CONFIANCA = 0.95
N_ZERO = n_minimo(TETO_FALSO_POSITIVO, 0, CONFIANCA)  # 119
# Limite computacional congelado: cinco vezes o plano k=0. Permite chegar ao
# checkpoint de k=8 (n=573); se os falsos positivos exigirem mais, a família
# reprova em vez de transformar a calibração numa busca sem limite.
TETO_SERIES_POR_FAMILIA = 600
SEMENTE_BASE = 900_000
FAMILIAS = tuple(FamiliaNula)


@dataclass(frozen=True, slots=True)
class ConfiguracaoGarimpo:
    variante: str
    risco_por_operacao: float

    def __post_init__(self) -> None:
        from titan.aprendizagem import RISCOS, VARIANTES

        if self.variante not in VARIANTES:
            raise ValueError(f"variante deve ser uma de {VARIANTES}")
        if self.risco_por_operacao not in RISCOS:
            raise ValueError(f"risco deve ser um de {RISCOS}")


@dataclass(frozen=True, slots=True)
class DiagnosticoExecucao:
    sinais: int
    posicoes_abertas_normal: int
    posicoes_abertas_pessimista: int
    recusas_normal: Mapping[str, int]
    recusas_pessimista: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ResultadoAto:
    veredito: VereditoPortao
    diagnosticos: Mapping[str, DiagnosticoExecucao]


@dataclass(frozen=True, slots=True)
class ResumoFamilia:
    familia: str
    n: int
    k: int
    limite_cp95: float
    certificado: bool
    motivo: str
    primeira_semente: int
    ultima_semente: int


Sentinela = Mapping[str, Sequence[bool]] | Callable[[MundoF29, str], Sequence[bool]]
ExecutorAto = Callable[
    [MundoF29, ConfiguracaoGarimpo, ParamsPortao, Sentinela], ResultadoAto
]
Publicador = Callable[[Mapping[str, Any]], None]


def _partidas(fonte: Sentinela, mundo: MundoF29, candidato: str) -> tuple[bool, ...]:
    valores = fonte(mundo, candidato) if callable(fonte) else fonte[candidato]
    saida = tuple(bool(v) for v in valores)
    if not saida:
        raise ValueError(f"sentinela sem partidas para {candidato}")
    return saida


def _intencoes_nucleo(mundo: MundoF29) -> list[Intencao]:
    """Executa o Núcleo somente nas fronteiras de entrada do pregão à vista."""
    from titan.regras import Nucleo, PREGOES, janela_nucleo
    from titan.tipos_f2 import Campanha
    from titan.visao import VisaoMemoria, de_ns

    simbolos = tuple(s for s in PREGOES if s in mundo.m15)
    if set(simbolos) != set(PREGOES):
        raise ValueError("mundo F2-9 sem todos os índices do Núcleo")
    dados = {(s, "M15"): mundo.m15[s] for s in simbolos}
    entradas: set[datetime] = set()
    for s in simbolos:
        dias_locais = {de_ns(int(t)).astimezone(ZoneInfo(PREGOES[s].fuso)).date()
                       for t in mundo.m15[s].ts}
        entradas.update(janela_nucleo(s, d)[1] for d in dias_locais)
    if not entradas:
        return []
    visao = VisaoMemoria(
        dados, min(entradas), mundo.sessoes,
        universo={Campanha.DAY_TRADE_M15: simbolos},
    )
    regra = Nucleo(simbolos=simbolos)
    unicas: dict[str, Intencao] = {}
    for instante in sorted(entradas):
        for intencao in regra.decidir(visao.em(instante), instante):
            unicas[intencao.intencao_id] = intencao
    return sorted(unicas.values(), key=lambda i: (i.barra_sinal_fim, i.intencao_id))


def _rodar(
    mundo: MundoF29, intencoes: Sequence[Intencao], risco: float,
    pessimista: bool,
) -> ResultadoMotor:
    from titan.cadeia_garimpo import params_motor
    from titan.motor import simular

    simbolos = {i.simbolo for i in intencoes}
    carregador = {s: mundo.barras_motor(s) for s in simbolos}
    return simular(
        list(intencoes), carregador, mundo.specs, mundo.calendario,
        params_motor(risco, pessimista), somente_series=False,
    )


def _diagnostico(
    sinais: Sequence[Intencao], normal: ResultadoMotor,
    pessimista: ResultadoMotor,
) -> DiagnosticoExecucao:
    def aberturas(r: ResultadoMotor) -> int:
        return sum(x.get("evento") == "abertura" for x in r.livro)

    def recusas(r: ResultadoMotor) -> dict[str, int]:
        return dict(sorted(Counter(x.motivo.value for x in r.recusas).items()))

    return DiagnosticoExecucao(
        len(sinais), aberturas(normal), aberturas(pessimista),
        recusas(normal), recusas(pessimista),
    )


def executar_ato(
    mundo: MundoF29,
    configuracao: ConfiguracaoGarimpo,
    params_portao: ParamsPortao,
    partidas_sentinela: Sentinela,
) -> ResultadoAto:
    """Roda Núcleo + Garimpo, normal + pessimista, e um único Holm."""
    from titan import cadeia_garimpo as cg
    from titan.aprendizagem import agenda, aplicar
    from titan.prova import testar

    risco = configuracao.risco_por_operacao  # C3: Núcleo usa o vencedor também
    brutas = cg.intencoes(mundo)
    isoladas = cg.series_isoladas(mundo, brutas, risco)
    periodos = agenda(
        configuracao.variante, mundo.dias, isoladas,
        reamostragens=params_portao.reamostragens,
        semente=mundo.semente,
    )
    garimpo = aplicar([i for s in sorted(brutas) for i in brutas[s]], periodos)
    nucleo = _intencoes_nucleo(mundo)
    janelas = cg.janelas_padrao(mundo.dias)
    entradas: list[EntradaProva] = []
    diagnosticos: dict[str, DiagnosticoExecucao] = {}
    for nome, intencoes in (("Núcleo", nucleo), ("Garimpo", garimpo)):
        normal = _rodar(mundo, intencoes, risco, False)
        pessimista = _rodar(mundo, intencoes, risco, True)
        entradas.append(EntradaProva(
            nome, normal, pessimista, janelas.janela_inicio,
            janelas.cofre_inicio, _partidas(partidas_sentinela, mundo, nome),
        ))
        diagnosticos[nome] = _diagnostico(intencoes, normal, pessimista)
    # Cada mundo recebe sementes de bootstrap independentes e reprodutíveis.
    portao = replace(params_portao, semente=mundo.semente)
    return ResultadoAto(testar(entradas, portao), diagnosticos)


def _veredito_publicavel(v: VereditoPortao) -> dict[str, Any]:
    return {
        "ordem_holm": list(v.ordem),
        "candidatos": [asdict(c) for c in v.candidatos],
        "diagnostico_deriva": dict(v.diagnostico_deriva),
    }


class _JSONL:
    def __init__(self, caminho: Path):
        self.caminho = caminho
        caminho.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, item: Mapping[str, Any]) -> None:
        with self.caminho.open("a", encoding="utf-8") as arquivo:
            arquivo.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            arquivo.flush()


def projecao_custo(minutos_por_mundo: float = 35.0) -> dict[str, Any]:
    minimo = len(FAMILIAS) * N_ZERO
    maximo = len(FAMILIAS) * TETO_SERIES_POR_FAMILIA
    return {
        "familias": len(FAMILIAS),
        "mundos_minimos_k_zero": minimo,
        "mundos_maximos": maximo,
        "horas_minimas": round(minimo * minutos_por_mundo / 60, 1),
        "horas_no_teto": round(maximo * minutos_por_mundo / 60, 1),
        "memoria": "um mundo por processo; medir novamente antes de paralelizar",
        "observacao": (
            "35 min é o piloto de ato Garimpo anterior; Núcleo e o segundo "
            "candidato acrescentam custo, enquanto t30 ainda não foi medida ponta a ponta."
        ),
    }


def calibrar(
    sessoes: Mapping[str, Mapping[date, Sessao]],
    specs: Mapping[str, EspecificacaoAtivo],
    configuracao: ConfiguracaoGarimpo,
    params_portao: ParamsPortao,
    partidas_sentinela: Sentinela,
    *,
    executor: ExecutorAto = executar_ato,
    gerador: Callable[..., MundoF29] = gerar_mundo,
    familias: Sequence[FamiliaNula] = FAMILIAS,
    teto_series: int = TETO_SERIES_POR_FAMILIA,
    publicar: Publicador | None = None,
    saida_jsonl: Path | None = None,
) -> tuple[ResumoFamilia, ...]:
    """Certifica cada família por CP95 unilateral ou reprova no teto."""
    if teto_series < 1 or teto_series > TETO_SERIES_POR_FAMILIA:
        raise ValueError(f"teto_series deve estar em [1, {TETO_SERIES_POR_FAMILIA}]")
    if publicar is not None and saida_jsonl is not None:
        raise ValueError("use publicar ou saida_jsonl, não ambos")
    emissor = publicar or (_JSONL(saida_jsonl) if saida_jsonl is not None else None)
    resumos: list[ResumoFamilia] = []
    for familia in familias:
        familia = FamiliaNula(familia)
        # A faixa pertence à identidade da família, não à ordem/subconjunto
        # pedido numa retomada ou ensaio operacional.
        base = SEMENTE_BASE + FAMILIAS.index(familia) * TETO_SERIES_POR_FAMILIA
        n = k = 0
        certificado = False
        while n < teto_series:
            semente = base + n
            mundo = gerador(familia, semente, sessoes, specs)
            ato = executor(mundo, configuracao, params_portao, partidas_sentinela)
            aprovou = any(c.aprovado for c in ato.veredito.candidatos)
            n += 1
            k += int(aprovou)
            limite = limite_superior(k, n, CONFIANCA)
            # Evita procurar um n_minimo enorme quando já é matematicamente
            # impossível certificar dentro do teto computacional declarado.
            alvo_n = (
                teto_series + 1
                if limite_superior(k, teto_series, CONFIANCA)
                > TETO_FALSO_POSITIVO
                else n_minimo(TETO_FALSO_POSITIVO, k, CONFIANCA)
            )
            if emissor is not None:
                emissor({
                    "tipo": "serie", "familia": familia.value,
                    "semente": semente, "n": n, "k": k,
                    "aprovou_ato": aprovou, "limite_cp95": limite,
                    "proximo_checkpoint": alvo_n,
                    "veredito": _veredito_publicavel(ato.veredito),
                    "execucao": {
                        nome: asdict(d) for nome, d in ato.diagnosticos.items()
                    },
                })
            if n >= alvo_n and limite <= TETO_FALSO_POSITIVO:
                certificado = True
                break
        limite = limite_superior(k, n, CONFIANCA)
        resumo = ResumoFamilia(
            familia.value, n, k, limite, certificado,
            "cp95_dentro_de_2,5%" if certificado else "teto_sem_certificacao",
            base, base + n - 1,
        )
        resumos.append(resumo)
        if emissor is not None:
            emissor({"tipo": "familia", **asdict(resumo)})
    return tuple(resumos)


__all__ = [
    "CONFIANCA", "ConfiguracaoGarimpo", "DiagnosticoExecucao", "FAMILIAS",
    "N_ZERO", "ResultadoAto", "ResumoFamilia", "SEMENTE_BASE",
    "TETO_FALSO_POSITIVO", "TETO_SERIES_POR_FAMILIA", "calibrar",
    "executar_ato", "projecao_custo",
]
