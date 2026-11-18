from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from ferramentas import calibracao_f29 as c
from titan.motor import EspecificacaoAtivo
from titan.mundo_f29 import FamiliaNula
from titan.regras import PREGOES
from titan.tipos import Sessao
from titan.tipos_f2 import (
    Campanha, Intencao, Lado, ParamsPortao, ResultadoMotor,
    VereditoCandidato, VereditoPortao,
)


def _portao():
    return ParamsPortao(0.025, 0.8, 2, 30, 0.05, 0.2, 9, 77)


def _config():
    return c.ConfiguracaoGarimpo("V0", 0.005)


def _veredito(aprovado=False):
    candidatos = (
        VereditoCandidato("Núcleo", .5, .5, .5, .5, 30, 1., 0., .01,
                          aprovado, () if aprovado else ("p_acima_do_nivel",)),
        VereditoCandidato("Garimpo", .6, .6, .6, .6, 30, 1., 0., None,
                          False, ("holm_parou",)),
    )
    return VereditoPortao(("Núcleo", "Garimpo"), candidatos, {})


def _ato(aprovado=False):
    d = c.DiagnosticoExecucao(10, 3, 2, {"margem": 4}, {"margem": 5})
    return c.ResultadoAto(_veredito(aprovado), {"Núcleo": d, "Garimpo": d})


def _gerador_falso(familia, semente, sessoes, specs):
    return SimpleNamespace(familia=familia, semente=semente)


def _catalogo_pequeno():
    sessoes, specs = {}, {}
    dias = (date(2024, 1, 2), date(2024, 1, 3))
    for numero, simbolo in enumerate(PREGOES):
        por_dia = {}
        for dia in dias:
            abertura = datetime(dia.year, dia.month, dia.day, 9, tzinfo=UTC)
            por_dia[dia] = Sessao(
                simbolo, dia, abertura, abertura + timedelta(hours=1),
                abertura + timedelta(minutes=45), abertura + timedelta(days=1),
                0, False, "catálogo sintético do teste",
            )
        sessoes[simbolo] = por_dia
        specs[simbolo] = EspecificacaoAtivo(
            simbolo, "indice_spot", .01 + numero * .001,
            .1, .1, 10., .5, slippage_pontos=1,
            modo_contrato="indice_spot",
        )
    return sessoes, specs


def test_conta_e_projecao_congeladas():
    assert c.N_ZERO == 119
    p = c.projecao_custo()
    assert p["mundos_minimos_k_zero"] == 6 * 119
    assert p["mundos_maximos"] == 6 * 600
    assert p["horas_minimas"] == 416.5
    assert p["horas_no_teto"] == 2100.0


def test_k_zero_para_em_119_e_faixas_de_sementes_sao_disjuntas():
    vistos = []

    def executor(mundo, *_):
        vistos.append((mundo.familia, mundo.semente))
        return _ato(False)

    res = c.calibrar(
        {}, {}, _config(), _portao(), {"Núcleo": [False], "Garimpo": [False]},
        executor=executor, gerador=_gerador_falso,
    )
    assert len(res) == 6 and all(x.certificado and x.n == 119 and x.k == 0 for x in res)
    assert len(vistos) == 6 * 119
    assert [x.primeira_semente for x in res] == [900_000 + 600 * k for k in range(6)]
    assert len({semente for _, semente in vistos}) == len(vistos)


def test_um_falso_positivo_estende_ate_n_minimo_de_k_um():
    primeiro = True

    def executor(*_):
        nonlocal primeiro
        atual, primeiro = primeiro, False
        return _ato(atual)

    (r,) = c.calibrar(
        {}, {}, _config(), _portao(), {"Núcleo": [False], "Garimpo": [False]},
        executor=executor, gerador=_gerador_falso,
        familias=(FamiliaNula.AR_GARCH,),
    )
    assert (r.n, r.k, r.certificado) == (188, 1, True)


def test_teto_reprova_e_publica_aberturas_recusas_e_veredito():
    publicados = []
    (r,) = c.calibrar(
        {}, {}, _config(), _portao(), {"Núcleo": [False], "Garimpo": [False]},
        executor=lambda *_: _ato(True), gerador=_gerador_falso,
        familias=(FamiliaNula.GAPS,), teto_series=7,
        publicar=publicados.append,
    )
    assert (r.n, r.k, r.certificado, r.motivo) == (7, 7, False, "teto_sem_certificacao")
    primeira = publicados[0]
    assert primeira["tipo"] == "serie" and primeira["aprovou_ato"]
    assert primeira["execucao"]["Garimpo"]["posicoes_abertas_normal"] == 3
    assert primeira["execucao"]["Garimpo"]["recusas_pessimista"] == {"margem": 5}
    assert publicados[-1]["tipo"] == "familia"


def test_catalogo_sintetico_e_mundo_real_entram_por_injecao():
    sessoes, specs = _catalogo_pequeno()
    observados = []

    def executor(mundo, *_):
        observados.append(mundo)
        return _ato(False)

    (r,) = c.calibrar(
        sessoes, specs, _config(), _portao(),
        {"Núcleo": [False], "Garimpo": [False]}, executor=executor,
        familias=(FamiliaNula.FATORES_COMUNS,), teto_series=1,
    )
    assert not r.certificado and r.n == 1
    assert set(observados[0].m15) == set(PREGOES)
    assert observados[0].specs == specs and observados[0].semente == 901_800


def test_executar_ato_liga_dois_candidatos_e_sentinela(monkeypatch):
    from titan import aprendizagem, cadeia_garimpo, prova

    t = datetime(2024, 1, 2, 10, tzinfo=UTC)
    it_n = Intencao(Campanha.DAY_TRADE_M15, "N", "Usa500", Lado.COMPRA,
                    t, 10, t + timedelta(minutes=15))
    it_g = Intencao(Campanha.DAY_TRADE_M15, "G", "Usa500", Lado.VENDA,
                    t, 10, t + timedelta(minutes=15))
    vazio = ResultadoMotor((), (), (), (), (), (), (), (), (), ())
    mundo = SimpleNamespace(
        dias=[date(2023, 1, 2), date(2024, 1, 2)], semente=900_123,
    )
    monkeypatch.setattr(cadeia_garimpo, "intencoes", lambda w: {"Usa500": [it_g]})
    monkeypatch.setattr(cadeia_garimpo, "series_isoladas", lambda *a: {("G", "Usa500"): [0., 0.]})
    monkeypatch.setattr(cadeia_garimpo, "janelas_padrao", lambda d: SimpleNamespace(
        janela_inicio=d[0], cofre_inicio=d[1]))
    monkeypatch.setattr(aprendizagem, "agenda", lambda *a, **k: [object()])
    monkeypatch.setattr(aprendizagem, "aplicar", lambda *a: [it_g])
    monkeypatch.setattr(c, "_intencoes_nucleo", lambda w: [it_n])
    monkeypatch.setattr(c, "_rodar", lambda *a: vazio)
    monkeypatch.setattr(c, "_diagnostico", lambda *a: c.DiagnosticoExecucao(1, 0, 0, {}, {}))

    def testar(entradas, params):
        assert [e.candidato for e in entradas] == ["Núcleo", "Garimpo"]
        assert entradas[0].partidas_sentinela == (False, True)
        assert entradas[1].partidas_sentinela == (False,)
        assert params.semente == 900_123
        return _veredito(False)

    monkeypatch.setattr(prova, "testar", testar)
    r = c.executar_ato(
        mundo, _config(), _portao(),
        {"Núcleo": [False, True], "Garimpo": [False]},
    )
    assert set(r.diagnosticos) == {"Núcleo", "Garimpo"}


@pytest.mark.parametrize("variante,risco", [("VX", .005), ("V0", .007)])
def test_configuracao_vencedora_validada(variante, risco):
    with pytest.raises(ValueError):
        c.ConfiguracaoGarimpo(variante, risco)
