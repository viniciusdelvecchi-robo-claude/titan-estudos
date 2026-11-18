import json

import pytest

from ferramentas import calibrar_risco_aberto as cra
from titan.certificacao import limite_superior, n_minimo


def test_grade_cp_parada_monotona_sementes_e_saida(tmp_path):
    assert cra.GRADE_ALVOS == (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
    assert cra.N_MUNDOS == n_minimo(0.10, 0, 0.95) == 29
    chamadas = []

    def avaliar(semente, ms, alvo):
        chamadas.append((semente, tuple(ms), alvo))
        out = {}
        for m in ms:
            # M=50% reprova já em 5%; M=30%, em 30%; M=20% passa a grade inteira.
            falha = (m == 0.50 and alvo >= 0.05 or m == 0.30 and alvo >= 0.30) and semente == 140_000
            out[m] = cra.ResumoMundo(falha, 9, int(falha))
        return out

    path = tmp_path / "risco.json"
    doc = cra.calibrar("V2_20", 0.01, avaliar, saida=path)
    assert doc["alvo_por_m"] == {"0.20": 1.0, "0.30": 0.2, "0.50": None}
    pontos_50 = [p for p in doc["pontos"] if p["m"] == 0.50]
    pontos_30 = [p for p in doc["pontos"] if p["m"] == 0.30]
    assert len(pontos_50) == 1 and pontos_50[0]["aprovado"] is False
    assert [p["alvo"] for p in pontos_30] == [0.05, 0.10, 0.20, 0.30]
    assert pontos_30[-1]["k_mundos"] == 1
    assert pontos_30[-1]["limite_cp95"] == pytest.approx(limite_superior(1, 29, 0.95))
    assert not any(0.50 in ms and alvo > 0.05 for _, ms, alvo in chamadas)
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "completo"
    assert {s for s, _, _ in chamadas} == set(range(140_000, 140_029))


def test_validacao_e_projecao_nao_rodam_mundo():
    with pytest.raises(ValueError, match="variante"):
        cra.calibrar("", 0.01, lambda *_: {})
    with pytest.raises(ValueError, match="congelados"):
        cra.calibrar("V0", 0.03, lambda *_: {})
    custo = cra.projecao_custo()
    assert custo["mundos"] == 29
    assert custo["tarefas"] == 29
    assert custo["estimativa_linear_81_ativos_minutos_por_tarefa"] == pytest.approx(346.69)


def test_tarefas_agregadas_iguais_ao_calibrar_sequencial(tmp_path):
    assert cra.N_ATIVOS == 81
    assert cra.N_TAREFAS == 29
    linhas = []
    for tarefa in range(cra.N_TAREFAS):
        mundo_i = cra._coordenadas_tarefa(tarefa)
        semente_execucao = cra.SEMENTE_BASE + mundo_i
        semente_selecao = cra.SEMENTE_SELECAO_BASE + mundo_i
        por_alvo = {}
        for alvo in cra.GRADE_ALVOS:
            por_m = {}
            for m in cra.MS:
                falha = (m == 0.50 and alvo >= 0.05 or m == 0.30 and alvo >= 0.30) and mundo_i == 0
                por_m[f"{m:.2f}"] = {
                    "revogou_alguma": falha, "partidas": 9, "partidas_revogadas": int(falha),
                }
            por_alvo[f"{alvo:.2f}"] = por_m
        linhas.append(json.dumps({
            "tarefa": tarefa, "semente_execucao": semente_execucao,
            "semente_selecao": semente_selecao,
            "variante": cra.VARIANTE_OFICIAL, "risco": cra.RISCO_OFICIAL,
            "n_ativos": cra.N_ATIVOS, "por_alvo": por_alvo, "segundos": 1.0,
        }))
    (tmp_path / "tarefas.jsonl").write_text("\n".join(linhas) + "\n", encoding="utf-8")

    def avaliar(semente, ms, alvo):
        mundo_i = semente - cra.SEMENTE_BASE
        return {
            m: cra.ResumoMundo(
                (m == 0.50 and alvo >= 0.05 or m == 0.30 and alvo >= 0.30) and mundo_i == 0,
                9,
                int((m == 0.50 and alvo >= 0.05 or m == 0.30 and alvo >= 0.30) and mundo_i == 0),
            ) for m in ms
        }

    esperado = cra.calibrar(cra.VARIANTE_OFICIAL, cra.RISCO_OFICIAL, avaliar)
    obtido = cra.agregar_tarefas(tmp_path, tmp_path / "agregado.json")
    assert obtido == esperado


def test_preparacao_compacta_igual_a_objetos_campo_a_campo(monkeypatch):
    from titan import cadeia_garimpo as cg
    from titan.aprendizagem import agenda, aplicar
    from titan.sintetico import Vantagem, gerar_mundo

    monkeypatch.setattr(cra, "REAMOSTRAGENS", 19)
    semente, semente_selecao, n_dias, n_ativos = 771, 1771, 300, 1
    prep = cra._preparar_mundo(
        semente, "V3", 0.02, n_ativos=n_ativos, n_dias=n_dias,
        semente_selecao=semente_selecao,
    )
    selecao = gerar_mundo(
        semente_selecao, n_dias, n_ativos,
        Vantagem("momento", cra._forca_momento_sharpe_2(), 0.5),
    )
    execucao = gerar_mundo(semente, n_dias, n_ativos, Vantagem())
    brutas_selecao = cg.intencoes(selecao)
    series = cg.series_isoladas(selecao, brutas_selecao, 0.02)
    periodos = agenda("V3", selecao.dias, series, reamostragens=19, semente=semente_selecao)
    brutas_execucao = cg.intencoes(execucao)
    objetos = aplicar([i for s in sorted(brutas_execucao) for i in brutas_execucao[s]], periodos)
    compactas = [prep.lotes[s].materializar(k) for s in sorted(prep.lotes) for k in range(len(prep.lotes[s]))]
    assert compactas == objetos


def test_tarefa_grava_as_duas_sementes(monkeypatch, tmp_path):
    resumo = {m: cra.ResumoMundo(False, 10, 0) for m in cra.MS}
    monkeypatch.setattr(cra, "_preparar_mundo", lambda *args, **kwargs: object())
    monkeypatch.setattr(cra, "_avaliar_preparado", lambda *args, **kwargs: resumo)
    saida = tmp_path / "tarefa.jsonl"
    linha = cra.executar_tarefa(3, 1, saida)
    gravada = json.loads(saida.read_text(encoding="utf-8"))
    assert linha == gravada
    assert gravada["semente_execucao"] == 140_003
    assert gravada["semente_selecao"] == 240_003
