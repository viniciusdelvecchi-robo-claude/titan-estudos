"""Regressão do caminho compacto (gramática e aprendizagem) contra o caminho de objetos, campo a campo."""
import numpy as np

from titan import cadeia_garimpo as cg
from titan.aprendizagem import agenda, aplicar, aplicar_compacto
from titan.gramatica import gerar, gerar_compacto
from titan.sintetico import Vantagem, gerar_mundo


def _campos(i):
    return (i.intencao_id, i.campanha, i.regra_id, i.simbolo, i.lado, i.barra_sinal_fim, i.stop_pontos,
            i.hora_limite, i.alvo_pontos, i.peso_admissao)


def test_gerar_compacto_igual_a_gerar():
    w = gerar_mundo(41, 300, 2, Vantagem("sazonal", 0.1, 0.5))
    for s in w.m15:
        objs = gerar(s, w.m15[s], w.sessoes[s])
        lote = gerar_compacto(s, w.m15[s], w.sessoes[s])
        assert len(lote) == len(objs) > 1000
        assert [_campos(lote.materializar(k)) for k in range(len(lote))] == [_campos(i) for i in objs]


def test_aplicar_compacto_igual_a_aplicar():
    w = gerar_mundo(42, 420, 2, Vantagem("momento", 0.1, 0.5))
    its = cg.intencoes(w)
    ser = cg.series_isoladas(w, its, 0.01)
    lotes = {s: gerar_compacto(s, w.m15[s], w.sessoes[s]) for s in sorted(w.m15)}
    for v in ("V0", "V1", "V2_20"):
        per = agenda(v, w.dias, ser, reamostragens=99, semente=1)
        ref = aplicar([i for s in sorted(its) for i in its[s]], per)
        comp = aplicar_compacto(lotes, per)
        got = [_campos(l.materializar(k)) for s in sorted(comp) for l in [comp[s]] for k in range(len(l))]
        assert sorted(got, key=lambda x: (x[3], x[5], x[0])) == sorted(map(_campos, ref), key=lambda x: (x[3], x[5], x[0]))
        if v == "V1":
            assert any(x[-1] != 1.0 for x in got) and len(got) < len(ref) + 1
