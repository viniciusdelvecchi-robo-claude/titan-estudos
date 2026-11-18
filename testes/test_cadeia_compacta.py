"""A cadeia do Garimpo pelo caminho compacto dá exatamente as mesmas séries e a mesma avaliação."""
import numpy as np

from titan import cadeia_garimpo as cg
from titan.sintetico import Vantagem, gerar_mundo
from titan.tipos_f2 import ParamsPortao


def test_preparar_em_paralelo_igual_ao_sequencial():
    w = gerar_mundo(44, 300, 3, Vantagem("reversao", 0.09, 0.5))
    its_seq = cg.intencoes(w, compacto=True)
    ser_seq = {r: cg.series_isoladas(w, its_seq, r) for r in (0.005, 0.02)}
    lotes, ser_par = cg.preparar(w, (0.005, 0.02), processos=3)
    assert sorted(lotes) == sorted(its_seq)
    for s in lotes:
        assert len(lotes[s]) == len(its_seq[s])
        assert [lotes[s].intencao_id(k) for k in range(len(lotes[s]))] == \
            [its_seq[s].intencao_id(k) for k in range(len(its_seq[s]))]
    for r in ser_seq:
        assert list(ser_par[r]) == list(ser_seq[r])
        assert all(np.array_equal(ser_par[r][k], ser_seq[r][k]) for k in ser_seq[r])


def test_cadeia_compacta_igual_a_de_objetos():
    w = gerar_mundo(43, 420, 2, Vantagem("momento", 0.1, 0.5))
    its_o, its_c = cg.intencoes(w), cg.intencoes(w, compacto=True)
    ser_o, ser_c = cg.series_isoladas(w, its_o, 0.01), cg.series_isoladas(w, its_c, 0.01)
    assert sorted(ser_o) == sorted(ser_c)
    assert all(np.array_equal(ser_o[k], ser_c[k]) for k in ser_o)
    jan = cg.janelas_padrao(w.dias, 0.3)
    pp = ParamsPortao(0.025, 0.8, 2, 30, 0.05, 0.2, 99, 1)
    for v, r in (("V0", 0.005), ("V1", 0.02), ("V2_20", 0.01)):
        assert cg.avaliar_configuracao(w, its_o, ser_o, v, r, jan, pp) == \
            cg.avaliar_configuracao(w, its_c, ser_c, v, r, jan, pp)
