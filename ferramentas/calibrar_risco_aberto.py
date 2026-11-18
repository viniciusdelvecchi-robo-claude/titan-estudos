"""Calibração F2-10 do risco aberto alvo do sistema completo.

Uso (depois de F2-12 congelar a configuração)::

    python ferramentas/calibrar_risco_aberto.py V2 0.01

A saída é ``medidas/f2_10_risco_aberto.json``. Nada neste arquivo usa preço
real. O executor é deliberadamente de um processo: um mundo do tamanho oficial
já encosta no limite de memória desta máquina.

Unidade estatística
-------------------
As partidas mensais de um mesmo mundo se sobrepõem e não são binomiais
independentes. Por isso cada mundo é um ensaio CP: ele falha se *qualquer*
partida mensal completa de três anos revogar. Isso é conservador e certifica
um limite que também limita a chance de uma partida sorteada revogar.
"""

from __future__ import annotations

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
N_DIAS = 1_240
N_ATIVOS = 8
REAMOSTRAGENS = 499
PESSIMISTA = True
SAIDA_PADRAO = Path(__file__).resolve().parents[1] / "medidas" / "f2_10_risco_aberto.json"


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
    """Projeção antes da execução, sem fingir tempo ainda não medido."""
    cadeia_min_s = 86 + 180 + 65
    cadeia_max_s = 86 + 180 + 180
    pontos = len(GRADE_ALVOS)
    return {
        "mundos": N_MUNDOS,
        "pontos_maximos_grade": pontos,
        "cadeia_fixa_por_mundo_minutos": round(cadeia_min_s / 60, 2),
        "cadeia_fixa_por_mundo_max_minutos": round(cadeia_max_s / 60, 2),
        "cadeia_por_ponto_horas": round(N_MUNDOS * cadeia_min_s / 3600, 2),
        "cadeia_por_ponto_horas_max": round(N_MUNDOS * cadeia_max_s / 3600, 2),
        "teto_piloto_35min_por_ponto_horas": round(N_MUNDOS * 35 / 60, 2),
        "teto_piloto_grade_inteira_horas": round(N_MUNDOS * 35 * pontos / 60, 2),
        "observacao": (
            "Valores excluem as simulações mensais do sistema completo, ainda sem tempo medido. "
            "Um ponto custa 29 mundos; a parada monotônica evita os pontos posteriores ao primeiro reprovado."
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
        },
        "grade_alvos_fracao_orcamento_m_vezes_h": list(GRADE_ALVOS),
        "justificativa_grade": (
            "5% cobre a fronteira abaixo da capacidade prática; 10% a 100% em passos de 10 p.p. "
            "são números redondos, incluem o valor de trabalho de 50% e limitam o custo."
        ),
        "sementes": {"inicio": SEMENTE_BASE, "fim_inclusivo": SEMENTE_BASE + N_MUNDOS - 1},
        "garimpo": {"variante": variante, "risco_por_operacao": risco_por_operacao},
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
    intencoes_dia: list[tuple[object, object]]
    inicios: tuple
    dia_estado: object
    params_base: object


def _preparar_mundo(semente: int, variante: str, risco: float) -> _Preparado:
    # Imports tardios: importar o módulo e rodar o teste pequeno não materializa
    # mundo, gramática ou arrays do Garimpo.
    from ferramentas.viabilidade_f210 import _DiasEstados, _DiasIntencoes, _primeiros_pregoes_mensais, _somar_anos
    from titan import cadeia_garimpo as cg
    from titan.aprendizagem import agenda, aplicar
    from titan.sintetico import Vantagem, gerar_mundo

    mundo = gerar_mundo(semente, N_DIAS, N_ATIVOS, Vantagem())
    brutas = cg.intencoes(mundo)
    series = cg.series_isoladas(mundo, brutas, risco)
    periodos = agenda(variante, mundo.dias, series, reamostragens=REAMOSTRAGENS, semente=semente)
    todas = aplicar([i for s in sorted(brutas) for i in brutas[s]], periodos)
    janela = cg.janelas_padrao(mundo.dias)
    inicios = tuple(
        d for d in _primeiros_pregoes_mensais(mundo.dias, janela.janela_inicio, max(mundo.dias))
        if _somar_anos(d, 3) <= max(mundo.dias)
    )
    resolver = _DiasIntencoes(mundo)
    return _Preparado(
        mundo, [(i, resolver(i)) for i in todas], inicios, _DiasEstados(mundo),
        replace(cg.params_motor(risco, PESSIMISTA), capital_constante=False),
    )


def _avaliar_preparado(prep: _Preparado, ms: Sequence[float], alvo: float) -> Mapping[float, ResumoMundo]:
    from ferramentas.viabilidade_f210 import _somar_anos
    from titan.motor import simular
    from titan.tipos_f2 import EstadoRisco

    mundo = prep.mundo
    totais = {m: 0 for m in ms}
    revogadas = {m: 0 for m in ms}
    simbolos = sorted(mundo.m15)
    for inicio in prep.inicios:
        limite = _somar_anos(inicio, 3)
        its = [i for i, d in prep.intencoes_dia if inicio <= d < limite]
        carregador = {
            s: tuple(b for b in mundo.barras_motor(s) if b.data_servidor < limite)
            for s in simbolos
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


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("uso: python ferramentas/calibrar_risco_aberto.py VARIANTE RISCO_POR_OPERACAO")
    variante, risco = sys.argv[1], float(sys.argv[2])
    t0 = time.time()
    doc = executar_oficial(variante, risco)
    print(json.dumps({"saida": str(SAIDA_PADRAO), "alvo_por_m": doc["alvo_por_m"],
                      "horas": round((time.time() - t0) / 3600, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

