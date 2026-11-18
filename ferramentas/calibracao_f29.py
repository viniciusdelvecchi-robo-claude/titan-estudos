"""Executor da calibração F2-9 do ato inteiro em mundos sem vantagem.

O catálogo de sessões e contratos entra por injeção. Este módulo nunca abre
preços nem materializa o catálogo real; a porta do ensaio cego continua sendo
o único lugar em que os metadados persistidos são ligados ao ato.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
import json
import math
from pathlib import Path
import sys
import time
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
N_TAREFAS_INICIAIS = len(FAMILIAS) * N_ZERO
CATALOGO_PUBLICO = Path(__file__).resolve().parents[1] / "medidas" / "catalogo_f2_publico.json"
SAIDA_AGREGADA = Path(__file__).resolve().parents[1] / "medidas" / "f2_9_calibracao.json"
PILOTO_8_ATIVOS_SEGUNDOS = 153.34
ESTIMATIVA_LINEAR_81_ATIVOS_MINUTOS = round(PILOTO_8_ATIVOS_SEGUNDOS * 81 / 8 / 60, 2)


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


CONFIGURACAO_OFICIAL = ConfiguracaoGarimpo("V3", 0.02)
PORTAO_OFICIAL = ParamsPortao(0.025, 0.8, 2, 30, 0.05, 0.2, 499, SEMENTE_BASE)
SENTINELA_OFICIAL = {"Núcleo": (False,), "Garimpo": (False,)}


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


def _intencoes_nucleo(mundo: MundoF29, exigir_todos: bool = True) -> list[Intencao]:
    """Executa o Núcleo somente nas fronteiras de entrada do pregão à vista."""
    from titan.regras import Nucleo, PREGOES, janela_nucleo
    from titan.tipos_f2 import Campanha
    from titan.visao import VisaoMemoria, de_ns

    simbolos = tuple(s for s in PREGOES if s in mundo.m15)
    if exigir_todos and set(simbolos) != set(PREGOES):
        raise ValueError("mundo F2-9 sem todos os índices do Núcleo")
    if not simbolos:
        return []
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


def _rodar_par(
    mundo: MundoF29, intencoes, risco: float,
) -> tuple[ResultadoMotor, ResultadoMotor]:
    from titan.cadeia_garimpo import params_motor
    from titan.motor import simular_par

    lotes = getattr(intencoes, "lotes", None)
    simbolos = ({lote.simbolo for lote in lotes} if lotes is not None
                else {i.simbolo for i in intencoes})
    carregador = {s: mundo.barras_motor(s) for s in simbolos}
    return simular_par(
        intencoes, carregador, mundo.specs, mundo.calendario,
        params_motor(risco, False), params_motor(risco, True),
        somente_series=False,
    )


def _preparar_garimpo(
    mundo: MundoF29, configuracao: ConfiguracaoGarimpo,
    params_portao: ParamsPortao, processos: int,
):
    from titan import cadeia_garimpo as cg
    from titan.aprendizagem import agenda, aplicar_compacto
    from titan.intencoes_compactas import ColecaoIntencoesCompactas

    risco = configuracao.risco_por_operacao
    brutas, por_risco = cg.preparar(mundo, (risco,), processos=processos)
    periodos = agenda(
        configuracao.variante, mundo.dias, por_risco[risco],
        reamostragens=params_portao.reamostragens, semente=mundo.semente,
    )
    lotes = aplicar_compacto(brutas, periodos)
    return ColecaoIntencoesCompactas(lotes[s] for s in sorted(lotes) if len(lotes[s]))


def _diagnostico(
    sinais, normal: ResultadoMotor,
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
    *,
    processos: int = 1,
    exigir_nucleo: bool = True,
) -> ResultadoAto:
    """Roda Núcleo + Garimpo, normal + pessimista, e um único Holm."""
    from titan import cadeia_garimpo as cg
    from titan.prova import testar

    risco = configuracao.risco_por_operacao  # C3: Núcleo usa o vencedor também
    garimpo = _preparar_garimpo(mundo, configuracao, params_portao, processos)
    nucleo = _intencoes_nucleo(mundo, exigir_nucleo)
    janelas = cg.janelas_padrao(mundo.dias)
    entradas: list[EntradaProva] = []
    diagnosticos: dict[str, DiagnosticoExecucao] = {}
    for nome, intencoes in (("Núcleo", nucleo), ("Garimpo", garimpo)):
        normal, pessimista = _rodar_par(mundo, intencoes, risco)
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


def _proximo_checkpoint(k: int, teto_series: int) -> int:
    if limite_superior(k, teto_series, CONFIANCA) > TETO_FALSO_POSITIVO:
        return teto_series + 1
    return n_minimo(TETO_FALSO_POSITIVO, k, CONFIANCA)


def projecao_custo(minutos_por_mundo: float = ESTIMATIVA_LINEAR_81_ATIVOS_MINUTOS) -> dict[str, Any]:
    minimo = len(FAMILIAS) * N_ZERO
    maximo = len(FAMILIAS) * TETO_SERIES_POR_FAMILIA
    return {
        "familias": len(FAMILIAS),
        "mundos_minimos_k_zero": minimo,
        "mundos_maximos": maximo,
        "horas_minimas": round(minimo * minutos_por_mundo / 60, 1),
        "horas_no_teto": round(maximo * minutos_por_mundo / 60, 1),
        "piloto_8_ativos_segundos": PILOTO_8_ATIVOS_SEGUNDOS,
        "estimativa_linear_81_ativos_minutos": ESTIMATIVA_LINEAR_81_ATIVOS_MINUTOS,
        "expectativa_github_2x_minutos": round(ESTIMATIVA_LINEAR_81_ATIVOS_MINUTOS / 2, 2),
        "memoria": "um mundo por processo; medir novamente antes de paralelizar",
        "observacao": (
            "Piloto local: catálogo + geração AR/GARCH + V3 compacto + Núcleo disponível + "
            "par normal/pessimista + prova. O piloto de 8 ativos contém só 8 dos 12 índices "
            "do Núcleo; a tarefa oficial exige os 12 e será medida no GitHub antes da rodada."
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
            alvo_n = _proximo_checkpoint(k, teto_series)
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


def _coordenadas_tarefa(tarefa: int) -> tuple[FamiliaNula, int, int]:
    if not 0 <= tarefa < len(FAMILIAS) * TETO_SERIES_POR_FAMILIA:
        raise ValueError("tarefa fora do teto congelado")
    n, familia_i = divmod(tarefa, len(FAMILIAS))
    familia = FAMILIAS[familia_i]
    semente = SEMENTE_BASE + familia_i * TETO_SERIES_POR_FAMILIA + n
    return familia, n, semente


def executar_tarefa(tarefa: int, proc_interno: int, saida: Path) -> dict[str, Any]:
    if proc_interno < 1:
        raise ValueError("proc-interno deve ser positivo")
    from ferramentas.catalogo_publico_f29 import carregar

    familia, n, semente = _coordenadas_tarefa(tarefa)
    sessoes, specs = carregar(CATALOGO_PUBLICO)
    if len(specs) != 81:
        raise ValueError(f"catálogo público deve ter 81 ativos; recebeu {len(specs)}")
    t0 = time.time()
    mundo = gerar_mundo(familia, semente, sessoes, specs)
    ato = executar_ato(
        mundo, CONFIGURACAO_OFICIAL, PORTAO_OFICIAL, SENTINELA_OFICIAL,
        processos=proc_interno,
    )
    linha = {
        "tarefa": tarefa, "familia": familia.value, "n": n, "semente": semente,
        "n_ativos": len(specs), "configuracao": asdict(CONFIGURACAO_OFICIAL),
        "aprovou_ato": any(c.aprovado for c in ato.veredito.candidatos),
        "veredito": _veredito_publicavel(ato.veredito),
        "execucao": {nome: asdict(d) for nome, d in ato.diagnosticos.items()},
        "segundos": round(time.time() - t0, 1),
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    tmp = saida.with_suffix(saida.suffix + ".tmp")
    tmp.write_text(json.dumps(linha, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(saida)
    return linha


def agregar_tarefas(pasta: Path, saida: Path = SAIDA_AGREGADA) -> dict[str, Any]:
    por_familia: dict[FamiliaNula, dict[int, dict]] = {f: {} for f in FAMILIAS}
    for path in sorted(Path(pasta).rglob("*.jsonl")):
        for texto in path.read_text(encoding="utf-8").splitlines():
            if not texto.strip():
                continue
            d = json.loads(texto)
            familia, n, semente = _coordenadas_tarefa(int(d["tarefa"]))
            esperado = (familia.value, n, semente, 81, asdict(CONFIGURACAO_OFICIAL))
            obtido = (d["familia"], d["n"], d["semente"], d["n_ativos"], d["configuracao"])
            if obtido != esperado:
                raise ValueError(f"cabeçalho divergente na tarefa {d['tarefa']}")
            if n in por_familia[familia]:
                raise ValueError(f"tarefa repetida para {familia.value}, n={n}")
            por_familia[familia][n] = d

    resumos, pendentes, seguir = [], [], []
    for familia_i, familia in enumerate(FAMILIAS):
        linhas = por_familia[familia]
        n = k = 0
        certificado = False
        while n < TETO_SERIES_POR_FAMILIA and n in linhas:
            k += int(bool(linhas[n]["aprovou_ato"]))
            n += 1
            limite = limite_superior(k, n, CONFIANCA)
            checkpoint = _proximo_checkpoint(k, TETO_SERIES_POR_FAMILIA)
            if n >= checkpoint and limite <= TETO_FALSO_POSITIVO:
                certificado = True
                break
        if certificado or n == TETO_SERIES_POR_FAMILIA:
            resumo = ResumoFamilia(
                familia.value, n, k, limite_superior(k, n, CONFIANCA), certificado,
                "cp95_dentro_de_2,5%" if certificado else "teto_sem_certificacao",
                SEMENTE_BASE + familia_i * TETO_SERIES_POR_FAMILIA,
                SEMENTE_BASE + familia_i * TETO_SERIES_POR_FAMILIA + n - 1,
            )
            resumos.append(asdict(resumo))
            continue
        checkpoint = min(_proximo_checkpoint(k, TETO_SERIES_POR_FAMILIA), TETO_SERIES_POR_FAMILIA)
        faltantes = [x for x in range(n, checkpoint) if x not in linhas]
        tarefas = [x * len(FAMILIAS) + familia_i for x in faltantes]
        seguir.extend(tarefas)
        pendentes.append({
            "familia": familia.value, "n": n, "k": k,
            "proximo_checkpoint": checkpoint, "tarefas": tarefas,
        })
    doc = {
        "configuracao": asdict(CONFIGURACAO_OFICIAL),
        "familias_concluidas": resumos,
        "familias_pendentes": pendentes,
        "tarefas_seguir": seguir,
        "completo": not pendentes,
        "historia_sinal_embaralhado": "fora: exigiria preço real",
    }
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
            p.error("--listar não aceita saída")
        print(N_TAREFAS_INICIAIS)
    elif a.tarefa is not None:
        if a.saida is None:
            p.error("--tarefa exige arquivo JSONL de saída")
        print(json.dumps(executar_tarefa(a.tarefa, a.proc_interno, a.saida), ensure_ascii=False))
    else:
        if a.saida is not None:
            p.error("--agregar grava somente na saída oficial")
        doc = agregar_tarefas(a.agregar)
        print(json.dumps({"saida": str(SAIDA_AGREGADA), "tarefas_seguir": doc["tarefas_seguir"]}))


__all__ = [
    "CONFIANCA", "ConfiguracaoGarimpo", "DiagnosticoExecucao", "FAMILIAS",
    "N_ZERO", "ResultadoAto", "ResumoFamilia", "SEMENTE_BASE",
    "TETO_FALSO_POSITIVO", "TETO_SERIES_POR_FAMILIA", "calibrar",
    "executar_ato", "projecao_custo",
]


if __name__ == "__main__":
    main()
