import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ferramentas"))

import medir_causalidade_cadeia as mc  # noqa: E402
from titan.aprendizagem import Periodo, pesos  # noqa: E402


def test_f2_2_cadeia_inteira_um_corte():
    out = mc.conferir(1, n_dias=300)
    assert out["falhas"] == [] and out["antes_do_inicio_inalterado"]


def test_f2_2_detector_pega_aprendizagem_que_olha_o_futuro(monkeypatch):
    def agenda_trapaceira(variante, dias, series, *, reamostragens, semente):
        chaves = sorted(series)
        m = np.vstack([series[k] for k in chaves])
        return [Periodo(dias[0], None, pesos(variante, chaves, m, reamostragens=reamostragens, semente=semente))]

    monkeypatch.setattr(mc, "agenda", agenda_trapaceira)
    out = mc.conferir(1, n_dias=300)
    assert out["falhas"]
