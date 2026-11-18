"""Calibração F2-10 do risco aberto alvo do sistema completo.

Uso (depois de F2-12 congelar a configuração)::

    python ferramentas/calibrar_risco_aberto.py --listar
    python ferramentas/calibrar_risco_aberto.py --tarefa 0 --proc-interno 4 artefato.jsonl
    python ferramentas/calibrar_risco_aberto.py --agregar pasta_de_artefatos

A saída agregada é ``medidas/f2_10_risco_aberto.json``. Nada neste arquivo usa
preço real. Cada tarefa prepara o mundo plantado de seleção e o mundo nulo de
execução, pode paralelizar os ativos e avalia os onze alvos para os três M.

Unidade estatística
-------------------
As partidas mensais de um mesmo mundo se sobrepõem e não são binomiais
independentes. Por isso cada mundo é um ensaio CP: ele falha se *qualquer*
partida mensal completa de três anos revogar. Isso é conservador e certifica
um limite que também limita a chance de uma partida sorteada revogar.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.certificacao import limite_superior, n_minimo
from titan.aprendizagem import RISCOS, VARIANTES


# Congelado antes da primeira execução oficial. Frações do orçamento M×H.
# 5% testa a fronteira em que até o lote mínimo pode não caber; de 10% a 100%,
# passos redondos de 10 p.p. incluem o valor de trabalho de 50% da seção 4.
GRADE_ALVOS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
MS = (0.20, 0.30, 0.50)
TETO_REVOGACAO = 0.10
CONFIANCA = 0.95
N_MUNDOS = n_minimo(TETO_REVOGACAO, k_admitido=0, confianca=CONFIANCA)
SEMENTE_BASE = 140_000
SEMENTE_SELECAO_BASE = 240_000
N_DIAS = 1_240
N_ATIVOS = 81
REAMOSTRAGENS = 499
PESSIMISTA = True
VARIANTE_OFICIAL = "V3"
RISCO_OFICIAL = 0.02
N_TAREFAS = N_MUNDOS
PILOTO_8_ATIVOS_PREPARO_SEGUNDOS = 323.49
PILOTO_8_ATIVOS_PARTIDAS_SEGUNDOS = 1_730.97
SAIDA_PADRAO = Path(__file__).resolve().parents[1] / "medidas" / "f2_10_risco_aberto.json"
FORCAS_PATH = Path(__file__).resolve().parents[1] / "medidas" / "f2_12_forcas_vantagem.json"


def _forca_momento_sharpe_2() -> float:
    d = json.loads(FORCAS_PATH.read_text(encoding="utf-8"))
    try:
        forca = float(d["forcas"]["momento"]["2.0"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("forca congelada de momento Sharpe 2 ausente ou invalida") from exc
    if not math.isfinite(forca) or forca <= 0:
        raise ValueError("forca congelada de momento Sharpe 2 deve ser positiva e finita")
    return forca


@dataclass(frozen=True, slots=True)
class ResumoMundo:
    revogou_alguma: bool
    partidas: int
    partidas_revogadas: int


@dataclass(frozen=True, slots=True)
class PontoGrade:
    m: float
    alvo: float
    k_mundos: int
    n_mundos: int
    limite_cp95: float
    aprovado: bool
    partidas: int
    partidas_revogadas: int


def projecao_custo() -> dict[str, float | int | str]:
    """Projeção linear do piloto local de 8 ativos, medido em 18/11/2026."""
    fator = N_ATIVOS / 8
    preparo = PILOTO_8_ATIVOS_PREPARO_SEGUNDOS * fator
    partidas = PILOTO_8_ATIVOS_PARTIDAS_SEGUNDOS * fator
    estimado = preparo + partidas
    return {
        "mundos": N_MUNDOS,
        "tarefas": N_TAREFAS,
        "alvos_por_tarefa": len(GRADE_ALVOS),
        "piloto_8_ativos_preparo_segundos": PILOTO_8_ATIVOS_PREPARO_SEGUNDOS,
        "piloto_8_ativos_partidas_11x3m_segundos": PILOTO_8_ATIVOS_PARTIDAS_SEGUNDOS,
        "piloto_8_ativos_total_segundos": (
            PILOTO_8_ATIVOS_PREPARO_SEGUNDOS + PILOTO_8_ATIVOS_PARTIDAS_SEGUNDOS
        ),
        "estimativa_linear_81_ativos_preparo_minutos": round(preparo / 60, 2),
        "estimativa_linear_81_ativos_partidas_minutos": round(partidas / 60, 2),
        "estimativa_linear_81_ativos_minutos_por_tarefa": round(estimado / 60, 2),
        "estimativa_linear_total_horas": round(N_TAREFAS * estimado / 3600, 2),
        "observacao": (
            "Piloto com seleção momento Sharpe 2 e execução nula: 152403 intenções após V3, "
            "10 partidas mensais x 3 M x 11 alvos. A extrapolação linear fica apenas 3,31 min "
            "abaixo do teto de 350 min; medir 81 ativos no GitHub é obrigatório antes do disparo."
        ),
    }


def _documento(
    variante: str,
    risco_por_operacao: float,
    pontos: Sequence[PontoGrade],
    alvos: Mapping[float, float | None],
    status: str,
) -> dict:
    return {
        "criterio": {
            "teto_revogacao": TETO_REVOGACAO,
            "confianca_unilateral": CONFIANCA,
            "unidade_cp": "mundo independente; falha se alguma partida mensal de 3 anos revoga",
            "n_mundos": N_MUNDOS,
            "n_calculado_por": "titan.certificacao.n_minimo(teto=0.10, k_admitido=0, confianca=0.95)",
            "pessimista": PESSIMISTA,
            "n_ativos": N_ATIVOS,
        },
        "grade_alvos_fracao_orcamento_m_vezes_h": list(GRADE_ALVOS),
        "justificativa_grade": (
            "5% cobre a fronteira abaixo da capacidade prática; 10% a 100% em passos de 10 p.p. "
            "são números redondos, incluem o valor de trabalho de 50% e limitam o custo."
        ),
        "sementes": {
            "execucao_nula": {"inicio": SEMENTE_BASE, "fim_inclusivo": SEMENTE_BASE + N_MUNDOS - 1},
            "selecao_momento_sharpe_2": {
                "inicio": SEMENTE_SELECAO_BASE,
                "fim_inclusivo": SEMENTE_SELECAO_BASE + N_MUNDOS - 1,
            },
        },
        "garimpo": {
            "variante": variante,
            "risco_por_operacao": risco_por_operacao,
            "selecao": {
                "mundo": "momento Sharpe ideal 2; força lida de medidas/f2_12_forcas_vantagem.json",
                "fracao_ativos": 0.5,
            },
            "execucao": "mundo nulo independente; todo P&L, risco aberto e revogação vêm dele",
        },
        "nucleo": {
            "incluido": False,
            "motivo": "entra apenas no F2-9",
            "vies_esperado": (
                "Omiti-lo tende a subestimar exposição/revogação e escolher alvo alto demais; "
                "a direção não é garantia matemática porque o P&L do Núcleo também pode sustentar o patrimônio."
            ),
        },
        "pontos": [p.__dict__ if hasattr(p, "__dict__") else {
            "m": p.m, "alvo": p.alvo, "k_mundos": p.k_mundos, "n_mundos": p.n_mundos,
            "limite_cp95": p.limite_cp95, "aprovado": p.aprovado,
            "partidas": p.partidas, "partidas_revogadas": p.partidas_revogadas,
        } for p in pontos],
        "alvo_por_m": {f"{m:.2f}": alvos[m] for m in MS},
        "projecao_custo": projecao_custo(),
        "status": status,
    }


def _gravar(path: Path, documento: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporario = path.with_suffix(path.suffix + ".tmp")
    temporario.write_text(json.dumps(documento, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    temporario.replace(path)


def calibrar(
    variante: str,
    risco_por_operacao: float,
    avaliar_mundo: Callable[[int, Sequence[float], float], Mapping[float, ResumoMundo]],
    *,
    saida: Path | None = None,
) -> dict:
    """Percorre a grade em ordem e para cada M no primeiro ponto reprovado.

    ``avaliar_mundo(semente, ms_ativos, alvo)`` prepara o Garimpo uma vez e
    devolve os resultados dos M ainda ativos. A mesma semente é reutilizada em
    todos os pontos (números aleatórios comuns), mas mundos distintos são os 29
    ensaios independentes do CP.
    """
    if variante not in VARIANTES:
        raise ValueError(f"variante do Garimpo deve ser uma de {VARIANTES}")
    if not math.isfinite(risco_por_operacao) or risco_por_operacao not in RISCOS:
        raise ValueError(f"risco por operação deve ser um dos valores congelados {RISCOS}")

    ativos = set(MS)
    ultimo_aprovado: dict[float, float | None] = {m: None for m in MS}
    anterior_k = {m: -1 for m in MS}
    pontos: list[PontoGrade] = []
    for alvo in GRADE_ALVOS:
        if not ativos:
            break
        por_m: dict[float, list[ResumoMundo]] = {m: [] for m in ativos}
        for numero in range(N_MUNDOS):
            semente = SEMENTE_BASE + numero
            res = avaliar_mundo(semente, tuple(sorted(ativos)), alvo)
            if set(res) != ativos:
                raise ValueError("avaliador não devolveu exatamente os M ativos")
            for m, resumo in res.items():
                por_m[m].append(resumo)

        reprovados = set()
        for m in sorted(ativos):
            resultados = por_m[m]
            k = sum(r.revogou_alguma for r in resultados)
            if k < anterior_k[m]:
                raise RuntimeError(f"revogação não monotônica observada para M={m}: {k} < {anterior_k[m]}")
            anterior_k[m] = k
            limite = limite_superior(k, N_MUNDOS, CONFIANCA)
            aprovado = limite <= TETO_REVOGACAO
            pontos.append(PontoGrade(
                m, alvo, k, N_MUNDOS, limite, aprovado,
                sum(r.partidas for r in resultados),
                sum(r.partidas_revogadas for r in resultados),
            ))
            if aprovado:
                ultimo_aprovado[m] = alvo
            else:
                reprovados.add(m)
        ativos -= reprovados
        if saida is not None:
            _gravar(saida, _documento(variante, risco_por_operacao, pontos, ultimo_aprovado, "em_andamento"))

    doc = _documento(variante, risco_por_operacao, pontos, ultimo_aprovado, "completo")
    if saida is not None:
        _gravar(saida, doc)
    return doc


@dataclass(slots=True)
class _Preparado:
    mundo: object
    lotes: Mapping[str, object]
    dias_ordinais: Mapping[str, object]
    inicios: tuple
    dia_estado: object
    params_base: object


def _dias_ordinais_lote(mundo, lote):
    """Dia de servidor de cada linha, em vetor compacto, sem criar ``Intencao``."""
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


def _preparar_mundo(
    semente: int, variante: str, risco: float, processos: int = 1, *,
    n_ativos: int = N_ATIVOS, n_dias: int = N_DIAS, semente_selecao: int | None = None,
) -> _Preparado:
    # Imports tardios: importar o módulo e rodar o teste pequeno não materializa
    # mundo, gramática ou arrays do Garimpo.
    from ferramentas.viabilidade_f210 import _DiasEstados, _primeiros_pregoes_mensais, _somar_anos
    from titan import cadeia_garimpo as cg
    from titan.aprendizagem import agenda, aplicar_compacto
    from titan.sintetico import Vantagem, gerar_mundo

    if semente_selecao is None:
        semente_selecao = SEMENTE_SELECAO_BASE + (semente - SEMENTE_BASE)
    selecao = gerar_mundo(
        semente_selecao, n_dias, n_ativos,
        Vantagem("momento", _forca_momento_sharpe_2(), 0.5),
    )
    brutas_selecao, por_risco = cg.preparar(selecao, (risco,), processos=processos)
    series = por_risco[risco]
    periodos = agenda(
        variante, selecao.dias, series,
        reamostragens=REAMOSTRAGENS, semente=semente_selecao,
    )
    simbolos_selecao = set(selecao.m15)
    lotes_selecao = set(brutas_selecao)
    del selecao, brutas_selecao, por_risco, series
    mundo = gerar_mundo(semente, n_dias, n_ativos, Vantagem())
    if simbolos_selecao != set(mundo.m15):
        raise ValueError("simbolos dos mundos de selecao e execucao divergem")
    brutas_execucao, _ = cg.preparar(mundo, (), processos=processos)
    if lotes_selecao != set(brutas_execucao):
        raise ValueError("lotes dos mundos de selecao e execucao divergem")
    lotes = aplicar_compacto(brutas_execucao, periodos)
    if sum(map(len, lotes.values())) == 0:
        raise RuntimeError("V3 nao emitiu intencoes no mundo nulo apos aplicar a agenda de selecao")
    janela = cg.janelas_padrao(mundo.dias)
    inicios = tuple(
        d for d in _primeiros_pregoes_mensais(mundo.dias, janela.janela_inicio, max(mundo.dias))
        if _somar_anos(d, 3) <= max(mundo.dias)
    )
    return _Preparado(
        mundo, lotes, {s: _dias_ordinais_lote(mundo, lote) for s, lote in lotes.items()},
        inicios, _DiasEstados(mundo),
        replace(cg.params_motor(risco, PESSIMISTA), capital_constante=False),
    )


def _avaliar_preparado(prep: _Preparado, ms: Sequence[float], alvo: float) -> Mapping[float, ResumoMundo]:
    import numpy as np
    from ferramentas.viabilidade_f210 import _somar_anos
    from titan.intencoes_compactas import ColecaoIntencoesCompactas
    from titan.motor import simular
    from titan.tipos_f2 import EstadoRisco

    mundo = prep.mundo
    totais = {m: 0 for m in ms}
    revogadas = {m: 0 for m in ms}
    simbolos = sorted(mundo.m15)
    for inicio in prep.inicios:
        limite = _somar_anos(inicio, 3)
        lotes = []
        for s in sorted(prep.lotes):
            d = prep.dias_ordinais[s]
            linhas = np.flatnonzero((d >= inicio.toordinal()) & (d < limite.toordinal())).astype(np.int64)
            if len(linhas):
                lotes.append(prep.lotes[s].selecionar(np.ascontiguousarray(linhas)))
        its = ColecaoIntencoesCompactas(lotes)
        simbolos_partida = {lote.simbolo for lote in lotes}
        carregador = {
            s: tuple(b for b in mundo.barras_motor(s) if b.data_servidor < limite)
            for s in simbolos if s in simbolos_partida
        }
        for m in ms:
            params = replace(prep.params_base, limiar_m=m, risco_aberto_alvo=alvo, capital_constante=False)
            resultado = simular(its, carregador, mundo.specs, mundo.calendario, params)
            revogou = any(
                e is EstadoRisco.REVOGADO and inicio <= prep.dia_estado(t) < limite
                for t, e in resultado.estados
            )
            totais[m] += 1
            revogadas[m] += int(revogou)
    return {
        m: ResumoMundo(revogadas[m] > 0, totais[m], revogadas[m])
        for m in ms
    }


def executar_oficial(variante: str, risco: float, saida: Path = SAIDA_PADRAO) -> dict:
    def avaliar(semente: int, ms: Sequence[float], alvo: float) -> Mapping[float, ResumoMundo]:
        # Os três M ativos compartilham o mesmo mundo e Garimpo. Ao retornar, a
        # única referência pesada morre; nunca há dois mundos simultâneos.
        prep = _preparar_mundo(semente, variante, risco)
        return _avaliar_preparado(prep, ms, alvo)

    return calibrar(variante, risco, avaliar, saida=saida)


def _coordenadas_tarefa(numero: int) -> int:
    if not 0 <= numero < N_TAREFAS:
        raise ValueError(f"tarefa deve estar entre 0 e {N_TAREFAS - 1}")
    return numero


def executar_tarefa(numero: int, proc_interno: int, saida: Path) -> dict:
    if proc_interno < 1:
        raise ValueError("proc-interno deve ser positivo")
    mundo_i = _coordenadas_tarefa(numero)
    semente = SEMENTE_BASE + mundo_i
    semente_selecao = SEMENTE_SELECAO_BASE + mundo_i
    t0 = time.time()
    prep = _preparar_mundo(
        semente, VARIANTE_OFICIAL, RISCO_OFICIAL, proc_interno,
        semente_selecao=semente_selecao,
    )
    por_alvo = {}
    for alvo in GRADE_ALVOS:
        por_m = _avaliar_preparado(prep, MS, alvo)
        por_alvo[f"{alvo:.2f}"] = {f"{m:.2f}": {
            "revogou_alguma": r.revogou_alguma, "partidas": r.partidas,
            "partidas_revogadas": r.partidas_revogadas,
        } for m, r in por_m.items()}
    linha = {
        "tarefa": numero, "semente_execucao": semente, "semente_selecao": semente_selecao,
        "variante": VARIANTE_OFICIAL, "risco": RISCO_OFICIAL, "n_ativos": N_ATIVOS,
        "por_alvo": por_alvo,
        "segundos": round(time.time() - t0, 1),
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    tmp = saida.with_suffix(saida.suffix + ".tmp")
    tmp.write_text(json.dumps(linha, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(saida)
    return linha


def agregar_tarefas(pasta: Path, saida: Path = SAIDA_PADRAO) -> dict:
    por_tarefa = {}
    for path in sorted(pasta.rglob("*.jsonl")):
        for bruto in path.read_text(encoding="utf-8").splitlines():
            if not bruto.strip():
                continue
            d = json.loads(bruto)
            numero = d["tarefa"]
            if numero in por_tarefa:
                raise ValueError(f"tarefa repetida: {numero}")
            mundo_i = _coordenadas_tarefa(numero)
            esperado = (
                SEMENTE_BASE + mundo_i, SEMENTE_SELECAO_BASE + mundo_i,
                VARIANTE_OFICIAL, RISCO_OFICIAL, N_ATIVOS,
            )
            obtido = (
                d["semente_execucao"], d["semente_selecao"],
                d["variante"], d["risco"], d["n_ativos"],
            )
            alvos_ok = set(d["por_alvo"]) == {f"{a:.2f}" for a in GRADE_ALVOS}
            ms_ok = all(set(x) == {f"{m:.2f}" for m in MS} for x in d["por_alvo"].values())
            if obtido != esperado or not alvos_ok or not ms_ok:
                raise ValueError(f"cabecalho divergente na tarefa {numero}")
            por_tarefa[numero] = d
    faltantes = sorted(set(range(N_TAREFAS)) - set(por_tarefa))
    if faltantes:
        raise ValueError(f"faltam {len(faltantes)} tarefas; primeira: {faltantes[0]}")

    def avaliar(semente: int, ms: Sequence[float], alvo: float):
        numero = semente - SEMENTE_BASE
        bruto = por_tarefa[numero]["por_alvo"][f"{alvo:.2f}"]
        return {m: ResumoMundo(**bruto[f"{m:.2f}"]) for m in ms}

    doc = calibrar(VARIANTE_OFICIAL, RISCO_OFICIAL, avaliar)
    _gravar(saida, doc)
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
        print(N_TAREFAS)
    elif a.tarefa is not None:
        if a.saida is None:
            p.error("--tarefa exige o arquivo JSONL de saida")
        print(json.dumps(executar_tarefa(a.tarefa, a.proc_interno, a.saida), ensure_ascii=False))
    else:
        if a.saida is not None:
            p.error("--agregar grava somente na saida oficial")
        doc = agregar_tarefas(a.agregar)
        print(json.dumps({"saida": str(SAIDA_PADRAO), "alvo_por_m": doc["alvo_por_m"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

