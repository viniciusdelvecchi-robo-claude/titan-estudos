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
    assert custo["teto_piloto_35min_por_ponto_horas"] == pytest.approx(16.92)
    assert custo["teto_piloto_grade_inteira_horas"] == pytest.approx(186.08)
