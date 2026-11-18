from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import math

import pytest

import ferramentas.viabilidade_f210 as f210
from titan.motor import BarraMotor, EspecificacaoAtivo
from titan.sintetico import Mundo
from titan.tipos import Sessao


def test_f2_6_cdi_publica_cambio_parado_e_historico_com_fontes_injetadas():
    fonte_cdi = f210.FonteCDI(0.10, "série CDI sintética", "teste conhecido")
    fonte_fx = f210.FonteCambio("USD/BRL sintético", "teste conhecido")
    pontos = [
        f210.PontoCDICambio(date(2024, 1, 2), 1.10, 5.0),
        f210.PontoCDICambio(date(2024, 1, 3), 1.10, 5.5),
    ]
    r = f210.comparar_cdi_cambios(3000.0, pontos, fonte_cdi, fonte_fx)
    # CDI bruto = 1,21; ganho líquido após 15% = 17,85%.
    assert r.multiplicador_cdi_bruto == pytest.approx(1.21)
    assert r.saldo_cdi_cambio_parado_usd == pytest.approx(3000 * 1.1785)
    assert r.saldo_cdi_cambio_historico_usd == pytest.approx(3000 * 5 / 5.5 * 1.1785)
    assert r.saldo_dolar_parado_usd == 3000.0
    assert r.fonte_cdi is fonte_cdi and r.fonte_cambio is fonte_fx
from titan.tipos_f2 import (
    Campanha, EstadoRisco, Intencao, Lado, ParamsMotor, ResultadoMotor,
)
from titan.visao import Barras


def _mundo_pequeno() -> tuple[Mundo, list[Intencao]]:
    # 49 pregões mensais em quatro anos: dois ativos e muito menos de 300 dias.
    dias = []
    for n in range(49):
        ano, mes0 = 2020 + n // 12, n % 12
        dias.append(date(ano, mes0 + 1, 2))
    m15, sessoes, specs, intencoes = {}, {}, {}, []
    for a in range(2):
        s = f"X{a}"
        ts, o, h, l, c, spread = [], [], [], [], [], []
        por_dia = {}
        for n, d in enumerate(dias):
            abertura = datetime(d.year, d.month, d.day, 10, tzinfo=UTC)
            ts.append(int(abertura.timestamp() * 1e9))
            o.append(1000 + n); h.append(1010 + n); l.append(990 + n); c.append(1001 + n); spread.append(1)
            por_dia[d] = Sessao(s, d, abertura, abertura + timedelta(hours=2),
                                  abertura + timedelta(minutes=30), abertura + timedelta(hours=14),
                                  0, False, "teste")
            intencoes.append(Intencao(Campanha.DAY_TRADE_M15, "R", s, Lado.COMPRA,
                                       abertura + timedelta(minutes=15), 10,
                                       abertura + timedelta(minutes=30), peso_admissao=2.0))
        import numpy as np
        m15[s] = Barras("M15", np.array(ts, dtype=np.int64), np.array(o), np.array(h),
                         np.array(l), np.array(c), np.array(spread))
        sessoes[s] = por_dia
        specs[s] = EspecificacaoAtivo(s, "cambio", 1.0, 0.1, 0.1, 10.0, 0.5)
    return Mundo(dias, m15, sessoes, specs, ()), intencoes


def _params() -> ParamsMotor:
    return ParamsMotor(3000.0, 0.01, 0.5, 0.2, 1 / 3, 0.5, 0.03, 0.2, 60,
                       False, capital_constante=False)


def _resultado_falso(intencoes, *_args, **_kwargs):
    inicio = min(i.barra_sinal_fim.date() for i in intencoes)
    datas = (inicio, inicio + timedelta(days=1), inicio + timedelta(days=370), inicio + timedelta(days=740))
    pnl = (100.0, -50.0, -100.0, 200.0)
    estados = ((datetime.combine(datas[1], datetime.min.time(), UTC), EstadoRisco.REDUCAO),)
    return ResultadoMotor(datas, pnl, tuple(x / 3000 for x in pnl), (0.0,) * 4, (0.0,) * 4,
                          (), (), (), estados, ())


def test_relatorio_lado_a_lado_degraus_cdi_e_partidas_mensais(monkeypatch):
    mundo, intencoes = _mundo_pequeno()
    monkeypatch.setattr(f210, "simular", _resultado_falso)
    fonte = f210.FonteCDI(0.0959, "Banco Central do Brasil", "série até 09/10/2025")
    r = f210.avaliar_viabilidade(mundo, intencoes, _params(),
                                 {0.2: 0.5, 0.3: 0.4, 0.5: 0.3}, fonte,
                                 sharpe_plantado=1.0)
    assert len(r.agregados) == 9
    assert {a.m for a in r.agregados} == {0.2, 0.3, 0.5}
    assert {a.degrau for a in r.agregados} == {1, 2, 3}
    assert all(p.um_ano.resultado_usd == 50.0 for p in r.partidas)
    assert all(p.tres_anos.resultado_usd == 150.0 for p in r.partidas)
    assert all(p.um_ano.dias_sem_capacidade == 1 for p in r.partidas)
    assert all(math.isfinite(p.tres_anos.sharpe_equilibrio_cdi) for p in r.partidas)
    assert r.projeto_para_aqui is False
    assert r.fonte_cdi.fonte == "Banco Central do Brasil"
    assert all(len(p.resultado_acumulado_por_mes_usd) == 36 for p in r.partidas)
    assert all(p.resultado_acumulado_por_mes_usd[0] == 50.0 for p in r.partidas)


def test_faixa_resultado_por_idade_separa_vantagem_e_usa_quantis_tipo_7(
    monkeypatch,
):
    mundo, intencoes = _mundo_pequeno()
    monkeypatch.setattr(f210, "simular", _resultado_falso)
    sem = f210.avaliar_viabilidade(
        mundo, intencoes, _params(),
        {0.2: 0.5, 0.3: 0.4, 0.5: 0.3},
        f210.FonteCDI(0.1, "fonte", "data"), sharpe_plantado=0.0,
    )
    com = replace(
        sem,
        partidas=tuple(
            replace(
                p,
                resultado_acumulado_por_mes_usd=tuple(
                    x + 100.0 for x in p.resultado_acumulado_por_mes_usd
                ),
            )
            for p in sem.partidas
        ),
        sharpe_plantado=1.0,
    )
    faixas = f210.faixas_resultado_por_idade([sem], [com])
    assert len(faixas) == 2 * 3 * 3 * 36
    base = next(
        x for x in faixas
        if x.cenario == "sem_vantagem" and x.m == 0.2
        and x.degrau == 1 and x.idade_meses == 1
    )
    vantagem = next(
        x for x in faixas
        if x.cenario == "com_vantagem" and x.m == 0.2
        and x.degrau == 1 and x.idade_meses == 1
    )
    assert (
        base.quantil_05_usd, base.quantil_25_usd, base.quantil_50_usd,
        base.quantil_75_usd, base.quantil_95_usd,
    ) == (50.0,) * 5
    assert vantagem.quantil_50_usd == 150.0


def test_quantil_tipo_7_interpola_e_faixa_exige_os_dois_cenarios():
    assert f210._quantil_tipo_7([0.0, 100.0], 0.25) == 25.0
    with pytest.raises(ValueError, match="com e sem vantagem"):
        f210.faixas_resultado_por_idade([], [])


def test_regra_de_parada_usa_degrau_inteiro_e_so_sharpe_um(monkeypatch):
    mundo, intencoes = _mundo_pequeno()

    def perda(intencoes, *_args, **_kwargs):
        inicio = min(i.barra_sinal_fim.date() for i in intencoes)
        return ResultadoMotor((inicio,), (-1.0,), (-1 / 3000,), (0.0,), (0.0,), (), (), (), (), ())

    monkeypatch.setattr(f210, "simular", perda)
    kwargs = dict(mundo=mundo, intencoes=intencoes, params_base=_params(),
                  risco_aberto_por_m={0.2: 0.5, 0.3: 0.5, 0.5: 0.5},
                  fonte_cdi=f210.FonteCDI(0.1, "fonte", "data"))
    assert f210.avaliar_viabilidade(**kwargs, sharpe_plantado=1.0).projeto_para_aqui is True
    assert f210.avaliar_viabilidade(**kwargs, sharpe_plantado=0.5).projeto_para_aqui is None


def test_exige_sistema_completo_alvos_calibrados_e_tres_anos():
    mundo, intencoes = _mundo_pequeno()
    fonte = f210.FonteCDI(0.1, "fonte", "data")
    with pytest.raises(ValueError, match="capital_constante=False"):
        f210.avaliar_viabilidade(mundo, intencoes, replace(_params(), capital_constante=True),
                                 {0.2: 0.5, 0.3: 0.5, 0.5: 0.5}, fonte)
    with pytest.raises(ValueError, match="exatamente"):
        f210.avaliar_viabilidade(mundo, intencoes, _params(), {0.2: 0.5}, fonte)
    curto = replace(mundo, dias=mundo.dias[:10])
    with pytest.raises(ValueError, match="três anos completos"):
        f210.avaliar_viabilidade(curto, intencoes[:20], _params(),
                                 {0.2: 0.5, 0.3: 0.5, 0.5: 0.5}, fonte)


def test_queda_maxima_e_revogacao_por_horizonte():
    fonte = f210.FonteCDI(0.1, "fonte", "data")
    inicio = date(2020, 1, 2)
    datas = (inicio, inicio + timedelta(days=1), inicio + timedelta(days=2))
    estados = ((datetime.combine(datas[2], datetime.min.time(), UTC), EstadoRisco.REVOGADO),)
    r = ResultadoMotor(datas, (300.0, -600.0, 100.0), (0.1, -0.2, 1 / 30),
                       (0.0,) * 3, (0.0,) * 3, (), (), (), estados, ())
    m = f210._metricas(r, inicio, inicio + timedelta(days=10), fonte, 3000.0)
    assert m.resultado_usd == -200.0
    assert m.queda_maxima_usd == 600.0
    assert m.queda_maxima_fracao == pytest.approx(600 / 3300)
    assert m.revogou and m.dias_sem_capacidade == 1


def test_tempo_sem_capacidade_usa_dia_do_servidor():
    from types import SimpleNamespace

    d = date(2024, 1, 2)
    abertura = datetime(2024, 1, 1, 21, tzinfo=UTC)
    sessao = Sessao("X", d, abertura, abertura + timedelta(days=1),
                    abertura + timedelta(hours=23), abertura + timedelta(days=1),
                    180, False, "teste")
    resolver = f210._DiasEstados(SimpleNamespace(sessoes={"X": {d: sessao}}))
    assert resolver(datetime(2024, 1, 1, 22, tzinfo=UTC)) == d
