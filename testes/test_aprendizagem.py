from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from titan.aprendizagem import (CONFIGURACOES, Periodo, agenda, aplicar, datas_reajuste, pesos_v1, pesos_v2,
                                sobreviventes_v3)
from titan.tipos_f2 import Campanha, Intencao, Lado


def _dias(inicio, fim):
    d, out = inicio, []
    while d < fim:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_quinze_configuracoes():
    assert len(CONFIGURACOES) == 15 and len(set(CONFIGURACOES)) == 15


def test_reajuste_primeiro_pregao_do_ano_com_doze_meses():
    dias = _dias(date(2021, 3, 1), date(2024, 6, 1))
    assert datas_reajuste(dias) == [date(2023, 1, 2), date(2024, 1, 1)]  # 2021 começa em março: incompleto
    dias2 = _dias(date(2021, 1, 4), date(2022, 6, 1))
    assert datas_reajuste(dias2) == [date(2022, 1, 3)]


def test_v1_e_v2():
    rng = np.random.default_rng(1)
    m = rng.normal(0, 0.01, (4, 500))
    m[0] += 0.003
    m[1] += 0.002
    m[3] = 0.0  # inelegível
    ch = [("B1", "A"), ("B1", "B"), ("B1", "C"), ("B1", "D")]
    w = pesos_v1(ch, m)
    assert w[ch[0]] > w[ch[1]] > 0 and w[ch[3]] == 0.0
    w2 = pesos_v2(ch, m, 2)
    assert w2 == {ch[0]: 1.0, ch[1]: 1.0, ch[2]: 0.0, ch[3]: 0.0}
    assert sum(pesos_v2(ch, m, 20).values()) == 3  # menos elegíveis que N: entram todos


def test_v2_empate_alfabetico():
    s = np.r_[np.full(10, 0.01), np.full(10, -0.005)]
    m = np.vstack([s, s])
    ch = [("B2", "Z"), ("B2", "A")]
    assert pesos_v2(ch, m, 1) == {("B2", "Z"): 0.0, ("B2", "A"): 1.0}


def test_v3_acha_o_sinal_e_controla_o_ruido():
    rng = np.random.default_rng(2)
    m = rng.normal(0, 0.01, (30, 750))
    m -= m.mean(axis=1, keepdims=True)  # ruído com média exatamente zero
    m[0] += 0.004
    ch = [("B3", f"X{i:02d}") for i in range(30)]
    s = sobreviventes_v3(ch, m, 299, 7)
    assert ("B3", "X00") in s and len(s) == 1


def _intencao(dia, regra="B1|sessao|f30h30", simbolo="EURUSD"):
    t = datetime(dia.year, dia.month, dia.day, 10, tzinfo=UTC)
    return Intencao(Campanha.DAY_TRADE_M15, regra, simbolo, Lado.COMPRA, t, 100, t + timedelta(hours=1))


def test_agenda_nao_olha_o_futuro_e_aplica():
    dias = _dias(date(2021, 1, 4), date(2023, 3, 1))
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, len(dias))
    b = rng.normal(0, 0.01, len(dias))
    ch_a, ch_b = ("B1|sessao|f30h30", "EURUSD"), ("B1|sessao|f30h30", "GBPUSD")
    base = agenda("V2_20", dias, {ch_a: a, ch_b: b}, reamostragens=99, semente=1)
    # alterar tudo a partir de 2023-01-02 não muda o peso do período que começa nesse dia
    a2, b2 = a.copy(), b.copy()
    corte = dias.index(date(2023, 1, 2))
    a2[corte:] = 9.0
    b2[corte:] = -9.0
    alt = agenda("V2_20", dias, {ch_a: a2, ch_b: b2}, reamostragens=99, semente=1)
    assert [p.pesos for p in base] == [p.pesos for p in alt]
    assert base[0].pesos is None and [p.inicio for p in base[1:]] == [date(2022, 1, 3), date(2023, 1, 2)]
    v1 = agenda("V1", dias, {ch_a: np.abs(a) + 0.001, ch_b: -np.abs(b) - 0.001}, reamostragens=99, semente=1)
    its = [_intencao(date(2021, 6, 1)), _intencao(date(2022, 6, 1)),
           _intencao(date(2022, 6, 1), simbolo="GBPUSD")]
    out = aplicar(its, v1)
    assert [i.simbolo for i in out] == ["EURUSD", "EURUSD"]  # 2021: pesos iguais; 2022: GBPUSD peso 0
    assert out[0].peso_admissao == 1.0 and out[1].peso_admissao > 1.0
    assert out[1].intencao_id == its[1].intencao_id


def test_reajuste_fora_de_janeiro_quando_janeiro_falta():
    dias = [d for d in _dias(date(2021, 1, 4), date(2022, 6, 1)) if not (date(2021, 12, 20) < d < date(2022, 2, 7))]
    assert datas_reajuste(dias) == [date(2022, 2, 7)]


def test_v1_variancia_zero_da_peso_infinito_e_admite():
    m = np.vstack([np.full(50, 0.001), np.r_[np.full(25, 0.001), np.full(25, -0.002)]])
    w = pesos_v1([("R", "A"), ("R", "B")], m)
    assert w[("R", "A")] == float("inf") and w[("R", "B")] == 0.0
    it = aplicar([_intencao(date(2022, 6, 1), regra="R", simbolo="A")],
                 [Periodo(date(2021, 1, 4), date(2022, 1, 3), None), Periodo(date(2022, 1, 3), None, w)])
    assert it[0].peso_admissao == float("inf")


def test_aplicar_usa_o_dia_do_servidor_na_virada_da_meia_noite():
    w = {("R", "A"): 0.0}
    per = [Periodo(date(2021, 1, 4), date(2022, 1, 3), None), Periodo(date(2022, 1, 3), None, w)]
    t = datetime(2022, 1, 2, 22, 30, tzinfo=UTC)  # 00:30 de 03/01 no servidor (UTC+2)
    i = Intencao(Campanha.DAY_TRADE_M15, "R", "A", Lado.COMPRA, t, 100, t + timedelta(hours=1))
    assert aplicar([i], per) == []  # pela data UTC (02/01) cairia nos pesos iguais e entraria


def test_aplicar_rapido_igual_a_referencia_literal():
    from dataclasses import replace
    from titan.aprendizagem import dia_servidor

    def aplicar_ref(its, periodos):  # a versão literal de C1, antes da otimização
        out = []
        for it in its:
            d = dia_servidor(it.barra_sinal_fim)
            per = next((p for p in reversed(periodos) if p.inicio <= d), periodos[0])
            w = 1.0 if per.pesos is None else float(per.pesos.get((it.regra_id, it.simbolo), 0.0))
            if w > 0:
                out.append(replace(it, peso_admissao=w))
        return out

    w = {("R", "A"): 2.5, ("R", "B"): 0.0, ("S", "A"): 1.0}
    per = [Periodo(date(2021, 1, 4), date(2022, 1, 3), None), Periodo(date(2022, 1, 3), date(2023, 1, 2), w),
           Periodo(date(2023, 1, 2), None, {("R", "A"): 0.5})]
    its = []
    for d in _dias(date(2020, 12, 1), date(2023, 3, 1))[::3]:
        for regra, sim in (("R", "A"), ("R", "B"), ("S", "A")):
            t = datetime(d.year, d.month, d.day, 21, 45, tzinfo=UTC)  # perto da virada do dia do servidor
            its.append(Intencao(Campanha.DAY_TRADE_M15, regra, sim, Lado.COMPRA, t, 100, t + timedelta(hours=1)))
    novo, ref = aplicar(its, per), aplicar_ref(its, per)
    assert [(i.intencao_id, i.peso_admissao) for i in novo] == [(i.intencao_id, i.peso_admissao) for i in ref]
    assert novo == ref and len(ref) > 200


def test_variante_desconhecida():
    with pytest.raises(ValueError):
        agenda("V9", _dias(date(2021, 1, 4), date(2022, 3, 1)), {("R", "A"): np.zeros(300)},
               reamostragens=9, semente=0)
