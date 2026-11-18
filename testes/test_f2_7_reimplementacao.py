"""F2-7: comparacao cega Codex x implementacao de referencia de Claude.

Este teste importa a referencia como caixa-preta; nao depende de detalhes
internos dela.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from titan.prova import testar as testar_referencia
from titan.prova_cega_codex import testar as testar_codex
from titan.tipos_f2 import EntradaProva, ParamsPortao, ResultadoMotor


N_DIAS = 1_200
N_PROVA = 1_000
B = 499
INICIO = date(2020, 1, 1)
DATAS = tuple(INICIO + timedelta(days=i) for i in range(N_DIAS))
COFRE_INICIO = DATAS[N_PROVA]


def _serie(semente: int, media: float, phi: float) -> np.ndarray:
    rng = np.random.default_rng(semente)
    inovacao = rng.normal(0.0, 0.01, N_DIAS)
    if phi == 0.0:
        return media + inovacao
    x = np.empty(N_DIAS, dtype=np.float64)
    x[0] = media + inovacao[0]
    escala = np.sqrt(1.0 - phi * phi)
    for i in range(1, N_DIAS):
        x[i] = media + phi * (x[i - 1] - media) + escala * inovacao[i]
    return x


def _resultado(retornos: np.ndarray, *, exposicao: float) -> ResultadoMotor:
    r = tuple(float(x) for x in retornos)
    pnl = tuple(x * 3000.0 for x in r)
    blocos = tuple(f"{d.isoformat()}|aposta" for d in DATAS)
    return ResultadoMotor(
        datas=DATAS,
        pnl_usd=pnl,
        retorno=r,
        exposicao_comprada=tuple(exposicao for _ in DATAS),
        exposicao_vendida=tuple(exposicao / 3.0 for _ in DATAS),
        blocos=blocos,
        livro=(),
        recusas=(),
        estados=(),
        conferencias_margem=(),
    )


def _candidato(nome: str, semente: int, media: float, phi: float) -> EntradaProva:
    normal = _serie(semente, media, phi)
    # Custo pessimista deterministico, sem trocar a dependencia serial.
    pessimista = normal - (0.00004 + (semente % 3) * 0.00001)
    return EntradaProva(
        candidato=nome,
        normal=_resultado(normal, exposicao=0.4 + media),
        pessimista=_resultado(pessimista, exposicao=0.4 + media),
        janela_inicio=DATAS[0],
        cofre_inicio=COFRE_INICIO,
        partidas_sentinela=(False,) * 24,
    )


def _params(semente: int) -> ParamsPortao:
    return ParamsPortao(
        fatia=0.025,
        folga=0.8,
        passos_holm=2,
        apostas_min=30,
        nivel_cofre=0.05,
        revogacao_max=0.20,
        reamostragens=B,
        semente=semente,
    )


def _comparar(real, cego) -> None:
    assert cego.ordem == real.ordem
    assert len(cego.candidatos) == len(real.candidatos)
    por_nome_real = {v.candidato: v for v in real.candidatos}
    por_nome_cega = {v.candidato: v for v in cego.candidatos}
    assert por_nome_cega.keys() == por_nome_real.keys()
    campos_p = (
        "p_newey_west",
        "p_bootstrap",
        "p_newey_west_pessimista",
        "p_bootstrap_pessimista",
        "p_cofre_negativo",
    )
    for nome in por_nome_real:
        esperado = por_nome_real[nome]
        obtido = por_nome_cega[nome]
        for campo in campos_p:
            assert getattr(obtido, campo) == pytest.approx(
                getattr(esperado, campo), abs=1e-12, rel=0.0
            ), (nome, campo)
        assert obtido.blocos_independentes == esperado.blocos_independentes
        assert obtido.nivel_aplicado == esperado.nivel_aplicado
        assert obtido.aprovado == esperado.aprovado
        assert obtido.motivos == esperado.motivos


@pytest.mark.parametrize("par", range(20))
def test_reimplementacao_cega_bate_campo_a_campo(par):
    media_a = 0.0015 * (par % 10) / 9.0
    media_b = 0.0015 * ((par * 7 + 3) % 10) / 9.0
    phi_a = 0.3 if par % 3 == 0 else 0.0
    phi_b = 0.3 if par % 4 == 0 else 0.0
    candidatos = (
        _candidato(f"A{par:02d}", 10_000 + 2 * par, media_a, phi_a),
        _candidato(f"B{par:02d}", 10_001 + 2 * par, media_b, phi_b),
    )
    params = _params(70_000 + par)
    real = testar_referencia(candidatos, params)
    cego = testar_codex(candidatos, params)
    _comparar(real, cego)
