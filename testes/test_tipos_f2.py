from datetime import UTC, date, datetime

import pytest

from titan.tipos_f2 import (Campanha, Intencao, Lado, ParamsMotor, ParamsPortao, ResultadoMotor)


def _params(**k):
    base = dict(capital_usd=3000.0, risco_por_operacao=0.02, risco_aberto_alvo=0.5, limiar_m=0.20,
                fracao_um_terco=1 / 3, margem_max=0.5, limite_perda_dia=0.03, filtro_spread_mov=0.2,
                filtro_janela_pregoes=60, pessimista=False)
    base.update(k)
    return ParamsMotor(**base)


def test_capital_diferente_do_petreo_falha():
    _params()
    with pytest.raises(ValueError):
        _params(capital_usd=30000.0)


def test_m_fora_das_opcoes_falha():
    with pytest.raises(ValueError):
        _params(limiar_m=0.25)


def _portao(fatia):
    return ParamsPortao(fatia=fatia, folga=0.8, passos_holm=2, apostas_min=30, nivel_cofre=0.05,
                        revogacao_max=0.2, reamostragens=999, semente=1)


def test_niveis_holm_com_folga_e_fatia_muda_critico():
    assert _portao(0.025).niveis() == pytest.approx((0.010, 0.020))
    assert _portao(0.0125).niveis() == pytest.approx((0.005, 0.010))


def test_portao_sem_fatia_falha():
    with pytest.raises(TypeError):
        ParamsPortao(folga=0.8, passos_holm=2, apostas_min=30, nivel_cofre=0.05,  # type: ignore[call-arg]
                     revogacao_max=0.2, reamostragens=999, semente=1)


def _int(**k):
    base = dict(campanha=Campanha.DAY_TRADE_M15, regra_id="r", simbolo="Usa500", lado=Lado.COMPRA,
                barra_sinal_fim=datetime(2024, 3, 1, 20, 0, tzinfo=UTC), stop_pontos=500,
                hora_limite=datetime(2024, 3, 1, 21, 45, tzinfo=UTC))
    base.update(k)
    return Intencao(**base)


def test_intencao_id_deterministico_e_sensivel():
    assert _int().intencao_id == _int().intencao_id
    assert _int().intencao_id != _int(stop_pontos=501).intencao_id
    assert _int(peso_admissao=2.0).intencao_id == _int(peso_admissao=1.0).intencao_id


def test_peso_admissao_invalido_falha():
    with pytest.raises(ValueError):
        _int(peso_admissao=float("nan"))
    with pytest.raises(ValueError):
        _int(peso_admissao=-1.0)
    assert _int(peso_admissao=float("inf")).peso_admissao == float("inf")


def test_hora_limite_fora_da_fronteira_de_15_min_falha():
    with pytest.raises(ValueError):
        _int(hora_limite=datetime(2024, 3, 1, 21, 50, tzinfo=UTC))


def test_resultado_exige_retorno_sobre_3000():
    d = (date(2024, 1, 2),)
    ResultadoMotor(d, (30.0,), (0.01,), (0.0,), (0.0,), (), (), (), (), ())
    with pytest.raises(ValueError):
        ResultadoMotor(d, (30.0,), (0.001,), (0.0,), (0.0,), (), (), (), (), ())
