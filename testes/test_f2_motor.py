from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest
import titan.motor as modulo_motor

from titan.motor import (
    AjusteMotor,
    BarraMotor,
    EspecificacaoAtivo,
    conferir_livro,
    simular,
    simular_isoladas,
    simular_par,
)
from titan.tipos import Sessao
from titan.tipos_f2 import (
    Campanha,
    EstadoRisco,
    Intencao,
    Lado,
    MotivoRecusa,
    ParamsMotor,
)


BASE = datetime(2024, 3, 1, 10, tzinfo=UTC)


def params(**mudancas):
    valores = dict(
        capital_usd=3000.0,
        risco_por_operacao=0.02,
        risco_aberto_alvo=0.5,
        limiar_m=0.20,
        fracao_um_terco=1 / 3,
        margem_max=0.5,
        limite_perda_dia=0.03,
        filtro_spread_mov=0.2,
        filtro_janela_pregoes=60,
        pessimista=False,
    )
    valores.update(mudancas)
    return ParamsMotor(**valores)


def barra(
    inicio,
    *,
    simbolo="X",
    tf="M15",
    o=1000,
    h=1010,
    l=995,
    c=1002,
    spread=1,
    caminho=(),
    apto=True,
    cambio=1.0,
):
    return BarraMotor(
        simbolo, inicio, tf, o, h, l, c, spread, inicio.date(), caminho, cambio, apto
    )


def historico(simbolo="X", *, spread=1, movimento=100):
    barras = []
    for n in range(60, 0, -1):
        t = BASE - timedelta(days=n)
        barras.append(
            barra(
                t,
                simbolo=simbolo,
                o=1000,
                h=1000 + movimento // 2,
                l=1000 - movimento // 2,
                c=1000,
                spread=spread,
            )
        )
    return barras


def sessao(simbolo, dia, *, limite=None):
    abertura = datetime.combine(dia, datetime.min.time(), UTC) + timedelta(hours=9)
    fechamento = abertura + timedelta(hours=14)
    limite = limite or abertura + timedelta(hours=12, minutes=45)
    return Sessao(
        simbolo,
        dia,
        abertura,
        fechamento,
        limite,
        datetime.combine(dia + timedelta(days=1), datetime.min.time(), UTC),
        0,
        False,
        "sintetico",
    )


def spec(simbolo="X", **mudancas):
    valores = dict(
        simbolo=simbolo,
        classe="cambio",
        valor_ponto_moeda_cotacao=1.0,
        lote_minimo=0.1,
        passo_lote=0.1,
        margem_por_lote_usd=10.0,
        stop_out_nivel_margem=0.5,
    )
    valores.update(mudancas)
    return EspecificacaoAtivo(**valores)


def intencao(
    *,
    simbolo="X",
    sinal=BASE,
    limite=None,
    stop=10,
    alvo=None,
    lado=Lado.COMPRA,
    campanha=Campanha.DAY_TRADE_M15,
    regra="r",
):
    return Intencao(
        campanha,
        regra,
        simbolo,
        lado,
        sinal,
        stop,
        limite or sinal.replace(hour=22, minute=0),
        alvo,
    )


def executar(
    atuais,
    intents,
    *,
    especificacoes=None,
    parametros=None,
    historicos=None,
    sessoes=None,
    cofre_inicio=None,
    lote_continuo_diagnostico=False,
    alarme=None,
):
    simbolos = {i.simbolo for i in intents}
    dados = {}
    for s in simbolos:
        dados[s] = list((historicos or {}).get(s, historico(s)))
        dados[s].extend(b for b in atuais if b.simbolo == s)
    contratos = especificacoes or {s: spec(s) for s in simbolos}
    calendario = dict(sessoes or {})
    for b in atuais:
        calendario.setdefault((b.simbolo, b.data_servidor), sessao(b.simbolo, b.data_servidor))
    return simular(
        intents, dados, contratos, calendario, parametros or params(),
        cofre_inicio=cofre_inicio,
        lote_continuo_diagnostico=lote_continuo_diagnostico,
        alarme=alarme,
    ), dados, contratos, calendario


def aberturas(resultado):
    return [r for r in resultado.livro if r["evento"] == "abertura"]


def fechamentos(resultado):
    return [r for r in resultado.livro if r["evento"] == "fechamento"]


def test_f2_1b_capital_constante_resultado_anterior_nao_muda_tamanho_posterior():
    d2 = BASE + timedelta(days=1)
    barras_a = [
        barra(BASE, h=1030, l=995, c=1025),
        barra(BASE.replace(hour=22), o=1025, h=1026, l=1024, c=1025),
        barra(d2, h=1030, l=995, c=1025),
        barra(d2.replace(hour=22), o=1025, h=1026, l=1024, c=1025),
    ]
    intents = [intencao(alvo=20), intencao(sinal=d2, limite=d2.replace(hour=22), alvo=20, regra="r2")]
    r1, _, _, _ = executar(barras_a, intents)
    # Primeiro dia agora perde; a ordem do segundo continua com o mesmo tamanho.
    barras_b = list(barras_a)
    barras_b[0] = barra(BASE, h=1005, l=980, c=985)
    r2, _, _, _ = executar(barras_b, intents)
    assert aberturas(r1)[1]["quantidade"] == aberturas(r2)[1]["quantidade"]


def test_f2_1b_resultado_de_um_mes_dez_vezes_maior_nao_muda_posicao_posterior():
    d2 = BASE + timedelta(days=32)
    atuais = [
        barra(BASE, h=1120, l=995, c=1100),
        barra(BASE.replace(hour=22), o=1100, h=1101, l=1099, c=1100),
        barra(d2, h=1010, l=995, c=1000),
        barra(d2.replace(hour=22), h=1005, l=995, c=1000),
    ]
    posterior = intencao(
        sinal=d2, limite=d2.replace(hour=22), alvo=5, regra="mes_seguinte"
    )
    pequeno, *_ = executar(
        atuais, [intencao(alvo=10, regra="mes_anterior"), posterior]
    )
    dez_vezes, *_ = executar(
        atuais, [intencao(alvo=100, regra="mes_anterior"), posterior]
    )
    assert pequeno.pnl_usd[0] * 10 == pytest.approx(dez_vezes.pnl_usd[0])
    assert aberturas(pequeno)[1]["quantidade"] == aberturas(dez_vezes)[1]["quantidade"]


def test_sistema_completo_dimensiona_dia_seguinte_pelo_patrimonio():
    d2 = BASE + timedelta(days=1)
    atuais = [
        barra(BASE, h=1200, l=995, c=1150),
        barra(BASE.replace(hour=22), o=1150, h=1151, l=1149, c=1150),
        barra(d2, h=1005, l=995, c=1000),
        barra(d2.replace(hour=22), h=1005, l=995, c=1000),
    ]
    intents = [
        intencao(alvo=100, regra="lucro"),
        intencao(sinal=d2, limite=d2.replace(hour=22), regra="seguinte"),
    ]
    constante, *_ = executar(atuais, intents)
    composto, *_ = executar(atuais, intents, parametros=params(capital_constante=False))
    assert aberturas(composto)[1]["quantidade"] > aberturas(constante)[1]["quantidade"]


def test_f2_3_entrada_so_na_barra_seguinte_e_preco_anterior_nao_dispara_stop():
    anterior = barra(BASE - timedelta(minutes=15), o=1000, h=1010, l=900, c=1000)
    entrada = barra(BASE, o=1000, h=1005, l=995, c=1002)
    limite = barra(BASE.replace(hour=22), o=1002, h=1003, l=1001, c=1002)
    r, *_ = executar([anterior, entrada, limite], [intencao()])
    assert aberturas(r)[0]["preco_entrada"] == 1001
    assert fechamentos(r)[0]["motivo"] == "zeragem"


def test_f2_3b_sem_caminho_stop_vence_alvo_e_com_caminho_ordem_real_vence():
    ambigua = barra(BASE, h=1030, l=980, c=1000)
    limite = barra(BASE.replace(hour=22), o=1000, h=1001, l=999, c=1000)
    r_pior, *_ = executar([ambigua, limite], [intencao(alvo=20)])
    assert fechamentos(r_pior)[0]["motivo"] == "stop"
    detalhada = barra(BASE, h=1030, l=980, c=1000, caminho=(1005, 1025, 985))
    r_real, *_ = executar([detalhada, limite], [intencao(alvo=20)])
    assert fechamentos(r_real)[0]["motivo"] == "alvo"


def test_f2_3c_sem_m15_inapto_e_diaria_d1_recebe_custo_da_convencao():
    so_h1 = barra(BASE, tf="H1", h=1010, l=990, c=1000)
    r, *_ = executar([so_h1], [intencao()])
    assert r.recusas[0].motivo is MotivoRecusa.INAPTO

    d1 = barra(BASE + timedelta(days=1), tf="D1", h=1010, l=995, c=1000)
    i = intencao(campanha=Campanha.DIARIA, limite=(BASE + timedelta(days=2)).replace(hour=22))
    sp = spec(custo_convencao_diaria_usd_por_lote=2.0)
    hist_d1 = [replace(b, tempo_grafico="D1") for b in historico()]
    r2, *_ = executar([d1], [i], especificacoes={"X": sp}, historicos={"X": hist_d1})
    assert aberturas(r2)[0]["fallback_diario"] is True
    sem_custo, *_ = executar(
        [d1], [i], especificacoes={"X": spec()}, historicos={"X": hist_d1}
    )
    assert r2.pnl_usd[-1] - sem_custo.pnl_usd[-1] == pytest.approx(
        -2.0 * aberturas(r2)[0]["quantidade"]
    )


def test_d0a_barra_inapta_recusa_entrada_mas_marca_e_fecha_posicao_aberta():
    entrada = barra(BASE, o=1000, h=1005, l=995, c=1000)
    saida_inapta = barra(
        BASE + timedelta(minutes=15), o=1000, h=1001, l=970, c=980,
        apto=False,
    )
    primeira = intencao(stop=20, regra="aberta_antes")
    segunda = intencao(
        sinal=BASE + timedelta(minutes=15), stop=20,
        limite=BASE.replace(hour=22), regra="entrada_inapta",
    )
    r, *_ = executar([entrada, saida_inapta], [primeira, segunda])
    assert len(aberturas(r)) == 1
    assert r.recusas[0].intencao_id == segunda.intencao_id
    assert r.recusas[0].motivo is MotivoRecusa.INAPTO
    fechamento = fechamentos(r)[0]
    assert fechamento["motivo"] == "stop"
    assert fechamento["instante"] == saida_inapta.inicio


@pytest.mark.parametrize("pessimista,extra_spread", [(False, 0), (True, 7)])
def test_d0b_cofre_sem_m15_fecha_no_stop_no_primeiro_dia_faltante(
    pessimista, extra_spread,
):
    primeiro_faltante = (BASE + timedelta(days=1)).date()
    segundo_faltante = (BASE + timedelta(days=2)).date()

    def sessao_faltante(dia):
        abertura = datetime.combine(dia, datetime.min.time(), UTC) + timedelta(hours=10)
        return Sessao(
            "X", dia, abertura, abertura + timedelta(hours=2),
            abertura + timedelta(hours=1, minutes=45),
            datetime.combine(dia + timedelta(days=1), datetime.min.time(), UTC),
            0, False, "sessao aberta sem M15",
        )

    calendario = {
        ("X", BASE.date()): sessao("X", BASE.date()),
        ("X", primeiro_faltante): sessao_faltante(primeiro_faltante),
        ("X", segundo_faltante): sessao_faltante(segundo_faltante),
    }
    hist = historico(spread=3)
    hist[10] = replace(hist[10], spread=8)
    entrada = barra(BASE, o=1000, h=1005, l=995, c=1000, spread=1)
    limite = datetime.combine(
        segundo_faltante + timedelta(days=1), datetime.min.time(), UTC
    ).replace(hour=22)
    i = intencao(
        campanha=Campanha.DIARIA, limite=limite, stop=20,
        regra="carregada_ao_cofre",
    )
    r, *_ = executar(
        [entrada], [i], historicos={"X": hist}, sessoes=calendario,
        especificacoes={"X": spec(slippage_pontos=2)},
        parametros=params(pessimista=pessimista),
        cofre_inicio=primeiro_faltante,
    )
    abertura = aberturas(r)[0]
    fechamento = fechamentos(r)[0]
    assert fechamento["motivo"] == "cofre_sem_m15_stop"
    assert fechamento["instante"] == calendario[("X", primeiro_faltante)].abertura_utc
    assert fechamento["preco_saida"] == abertura["stop"] - 2 - extra_spread
    assert primeiro_faltante in r.datas
    assert segundo_faltante not in r.datas
    assert r.pnl_usd[r.datas.index(primeiro_faltante)] == pytest.approx(
        fechamento["pnl_usd"]
    )


def test_cofre_com_m15_presente_preserva_resultado_campo_a_campo():
    atuais = [
        barra(BASE, o=1000, h=1005, l=995, c=1000),
        barra(BASE.replace(hour=22), o=1000, h=1001, l=999, c=1000),
    ]
    its = [intencao(stop=20)]
    atual, dados, contratos, calendario = executar(atuais, its)
    com_regra_d0b = simular(
        its, dados, contratos, calendario, params(),
        cofre_inicio=BASE.date(),
    )
    assert com_regra_d0b == atual


def test_f2_4_gap_abaixo_do_piso_revoga_e_estado_e_persistente():
    entrada = barra(BASE, h=1005, l=995, c=1000)
    gap = barra(BASE + timedelta(minutes=15), o=700, h=705, l=690, c=700)
    futura = intencao(sinal=BASE + timedelta(minutes=15), regra="depois")
    r, *_ = executar([entrada, gap], [intencao(), futura])
    assert any(e is EstadoRisco.REVOGADO for _, e in r.estados)
    assert any(x.motivo is MotivoRecusa.REVOGADO for x in r.recusas)


def test_f2_4_sem_posicao_e_folga_menor_que_lote_minimo_revoga():
    entrada = barra(BASE, h=1005, l=995, c=1000)
    # Gap realiza cerca de US$ 550: E ainda fica acima de F=2400, mas a
    # folga restante nao comporta a menor posicao (US$ 100).
    gap = barra(BASE + timedelta(minutes=15), o=915, h=916, l=910, c=915)
    r, *_ = executar(
        [entrada, gap], [intencao()],
        especificacoes={"X": spec(stop_minimo_pontos=1000)},
    )
    assert any(e is EstadoRisco.REVOGADO for _, e in r.estados)


def test_f2_4_exposicao_desconhecida_bloqueia_entrada_ate_reconciliar():
    class Carregador:
        def __init__(self, barras):
            self._barras = barras

        def barras_motor(self, simbolo):
            return self._barras[simbolo]

        def exposicao_conhecida(self, instante):
            return instante != BASE

    atuais = [barra(BASE), barra(BASE + timedelta(minutes=15))]
    dados = {"X": historico() + atuais}
    cal = {("X", b.data_servidor): sessao("X", b.data_servidor) for b in atuais}
    r = simular([intencao()], Carregador(dados), {"X": spec()}, cal, params())
    assert any(e is EstadoRisco.DESCONHECIDA for _, e in r.estados)
    assert r.recusas[0].motivo is MotivoRecusa.EXPOSICAO_DESCONHECIDA


def test_f2_4_reducao_so_termina_quando_l_volta_ao_piso():
    # Duas posicoes pequenas; alta eleva H e o recuo faz L < F. O motor fecha
    # contribuições em ordem ate restaurar a desigualdade, sem liberar antes.
    p = params(
        risco_por_operacao=0.005, limite_perda_dia=0.03, capital_constante=False
    )
    b0 = barra(BASE, h=1005, l=995, c=1000)
    b1 = barra(BASE + timedelta(minutes=15), o=8500, h=8505, l=8495, c=8500)
    b2 = barra(BASE + timedelta(minutes=30), o=1020, h=1025, l=1015, c=1020)
    intents = [intencao(stop=100, regra="a"), intencao(stop=100, regra="b")]
    r, *_ = executar([b0, b1, b2], intents, parametros=p)
    reducoes = [x for x in fechamentos(r) if x["motivo"] == "reducao_risco"]
    assert reducoes
    # Toda fotografia posterior a REDUCAO e NORMAL de novo satisfaz a acao de redução.
    assert r.estados[-1][1] in {EstadoRisco.NORMAL, EstadoRisco.REVOGADO}


def test_f2_4_reducao_parcial_insuficiente_recusa_intencao_concorrente():
    t1 = BASE + timedelta(minutes=15)
    t2 = BASE + timedelta(minutes=30)
    simbolos = ("X", "Y", "Z", "W")
    atuais = []
    for s in simbolos:
        atuais.extend([
            barra(BASE, simbolo=s, o=1000, h=1005, l=995, c=1000),
            barra(t1, simbolo=s, o=10000, h=10005, l=9995, c=10000),
            barra(t2, simbolo=s, o=7000, h=7005, l=6995, c=7000),
        ])
    iniciais = [intencao(simbolo=s, stop=100, regra=f"abre_{s}") for s in simbolos[:3]]
    concorrente = intencao(
        simbolo="W", sinal=t1, limite=BASE.replace(hour=22),
        stop=100, regra="concorrente",
    )
    r, *_ = executar(
        atuais, iniciais + [concorrente],
        especificacoes={s: spec(s) for s in simbolos},
        parametros=params(risco_por_operacao=0.005, capital_constante=False),
        historicos={s: historico(s) for s in simbolos},
    )
    reducoes = [x for x in fechamentos(r) if x["motivo"] == "reducao_risco"]
    assert len(reducoes) >= 2  # uma reducao parcial ainda deixava L < F
    assert any(
        x.intencao_id == concorrente.intencao_id and x.motivo is MotivoRecusa.REDUCAO
        for x in r.recusas
    )
    assert not any(x["intencao_id"] == concorrente.intencao_id for x in aberturas(r))


def test_f2_4_revogacao_por_fim_de_folga_dispara_adaptador_de_alarme():
    class AlarmeFalso:
        def __init__(self):
            self.eventos = []

        def enviar(self, evento):
            self.eventos.append(evento)

    alarme = AlarmeFalso()
    entrada = barra(BASE, h=1005, l=995, c=1000)
    gap = barra(BASE + timedelta(minutes=15), o=915, h=916, l=910, c=915)
    r, *_ = executar(
        [entrada, gap], [intencao()],
        especificacoes={"X": spec(stop_minimo_pontos=1000)},
        parametros=params(capital_constante=False), alarme=alarme,
    )
    assert any(e is EstadoRisco.REVOGADO for _, e in r.estados)
    assert len(alarme.eventos) == 1
    assert alarme.eventos[0]["tipo"] == "revogacao"
    assert alarme.eventos[0]["origem"] == "fim_folga"


def test_f2_4b_margem_antes_depois_e_cada_barra_e_aumento_intradiario_reduz():
    mudanca = BASE + timedelta(minutes=15)
    sp = spec(mudancas_margem=((mudanca, 1000.0),))
    atuais = [
        barra(BASE, h=1005, l=995, c=1000),
        barra(mudanca, o=1000, h=1005, l=995, c=1000),
    ]
    r, *_ = executar(atuais, [intencao()], especificacoes={"X": sp})
    origens = {c["origem"] for c in r.conferencias_margem}
    assert {"barra_antes", "ordem_antes", "ordem_depois", "barra_depois"} <= origens
    assert any(x["motivo"] == "stop_out" for x in fechamentos(r))


def test_f2_4b_margem_excessiva_bloqueia_antes_da_ordem():
    r, *_ = executar([barra(BASE)], [intencao()], especificacoes={"X": spec(margem_por_lote_usd=1000)})
    assert not aberturas(r)
    assert r.recusas[0].motivo is MotivoRecusa.MARGEM


def test_f2_4b_taxa_de_margem_e_mutuamente_exclusiva_com_usd():
    dinamica = spec(margem_por_lote_usd=None, taxa_margem_notional=0.02)
    assert dinamica.margem_no_instante(BASE, 1000, 1.5) == pytest.approx(30.0)
    with pytest.raises(ValueError, match="exatamente uma"):
        spec(taxa_margem_notional=0.02)
    with pytest.raises(ValueError, match="exatamente uma"):
        spec(margem_por_lote_usd=None)


def test_f2_4b_taxa_recalcula_preco_e_cambio_em_cada_barra_e_equivale_referencia():
    segunda = BASE + timedelta(minutes=15)
    atuais = [
        barra(BASE, o=1000, h=1004, l=996, c=1000, cambio=1.0),
        barra(segunda, o=1200, h=1204, l=1196, c=1200, cambio=1.5),
    ]
    contrato = {
        "X": spec(
            margem_por_lote_usd=None,
            taxa_margem_notional=0.02,
            lote_minimo=0.1,
            passo_lote=0.1,
        )
    }
    otimizado, dados, contratos, calendario = executar(
        atuais, [intencao(stop=100)], especificacoes=contrato
    )
    referencia = simular(
        [intencao(stop=100)], dados, contratos, calendario, params(),
        _otimizar=False,
    )
    assert otimizado == referencia
    quantidade = aberturas(otimizado)[0]["quantidade"]
    segunda_antes = next(
        c for c in otimizado.conferencias_margem
        if c["instante"] == segunda and c["origem"] == "barra_antes"
    )
    assert segunda_antes["margem_usd"] == pytest.approx(
        0.02 * 1200 * 1.0 * 1.5 * quantidade
    )


def test_d4_9_lote_continuo_ignora_minimo_e_passo_so_no_diagnostico():
    entrada = barra(BASE, o=1000, h=1005, l=995, c=1000)
    limite = barra(BASE.replace(hour=22), o=1000, h=1001, l=999, c=1000)
    i = intencao(stop=100)
    contrato = {"X": spec(lote_minimo=1.0, passo_lote=1.0)}
    normal, dados, contratos, calendario = executar(
        [entrada, limite], [i], especificacoes=contrato,
    )
    diagnostico, *_ = executar(
        [entrada, limite], [i], especificacoes=contrato,
        lote_continuo_diagnostico=True,
    )
    assert not aberturas(normal)
    assert normal.recusas[0].motivo is MotivoRecusa.LOTE_MINIMO
    assert aberturas(diagnostico)[0]["quantidade"] == pytest.approx(0.6)
    assert aberturas(diagnostico)[0]["risco_usd"] == pytest.approx(60.0)

    explicito_normal = simular(
        [i], dados, contratos, calendario, params(),
        lote_continuo_diagnostico=False,
    )
    assert explicito_normal == normal


def test_d4_9_lote_continuo_nao_revoga_por_fim_da_folga_do_lote_minimo():
    atuais, intencoes = [], []
    for numero in range(10):
        t = BASE + timedelta(days=numero)
        atuais.extend([
            barra(t, h=1005, l=995, c=1000),
            barra(
                t + timedelta(minutes=15), o=1000, h=1001, l=980, c=990
            ),
        ])
        intencoes.append(intencao(
            sinal=t, limite=t.replace(hour=22), regra=f"perda_{numero}"
        ))
    contrato = {"X": spec(stop_minimo_pontos=1000)}
    parametros = params(capital_constante=False)
    normal, *_ = executar(
        atuais, intencoes, especificacoes=contrato, parametros=parametros
    )
    continuo, *_ = executar(
        atuais, intencoes, especificacoes=contrato, parametros=parametros,
        lote_continuo_diagnostico=True,
    )
    assert any(e is EstadoRisco.REVOGADO for _, e in normal.estados)
    assert all(e is not EstadoRisco.REVOGADO for _, e in continuo.estados)


def test_f2_4c_indices_operam_sem_passar_um_terco_do_orcamento():
    sp = spec(classe="indice_spot")
    r, *_ = executar(
        [barra(BASE), barra(BASE.replace(hour=22))], [intencao()], especificacoes={"X": sp}
    )
    assert aberturas(r)
    limite = params().fracao_um_terco * params().limiar_m * 3000
    assert aberturas(r)[0]["risco_usd"] <= limite


def test_f2_5_conferencia_e_por_igualdade_pega_meio_spread_e_stop_adulterado():
    atuais = [barra(BASE), barra(BASE.replace(hour=22))]
    i = intencao()
    r, dados, contratos, calendario = executar(atuais, [i])
    conferir_livro(r, [i], dados, contratos, calendario, params())
    livro = [dict(x) for x in r.livro]
    abertura = next(x for x in livro if x["evento"] == "abertura")
    abertura["preco_entrada"] += 0.5
    adulterado = replace(r, livro=tuple(livro))
    with pytest.raises(ValueError, match="preco_entrada"):
        conferir_livro(adulterado, [i], dados, contratos, calendario, params())
    livro2 = [dict(x) for x in r.livro]
    next(x for x in livro2 if x["evento"] == "abertura")["stop"] -= 20
    with pytest.raises(ValueError, match="stop"):
        conferir_livro(replace(r, livro=tuple(livro2)), [i], dados, contratos, calendario, params())


def test_f2_6_dividendo_e_swap_por_modo_de_contrato_e_day_trade_sem_swap():
    instante_ajuste = BASE + timedelta(minutes=15)
    ajustes = (
        AjusteMotor(instante_ajuste, "dividendo", 10.0),
        AjusteMotor(instante_ajuste, "swap", -2.0),
    )
    atuais = [
        barra(BASE, h=1005, l=995, c=1000),
        barra(instante_ajuste, h=1005, l=995, c=1000),
        barra(BASE.replace(hour=22), h=1005, l=995, c=1000),
    ]
    diaria = intencao(campanha=Campanha.DIARIA)
    r, *_ = executar(atuais, [diaria], especificacoes={"X": spec(modo_contrato="indice_spot", ajustes=ajustes)})
    q = aberturas(r)[0]["quantidade"]
    assert sum(x["valor_usd"] for x in r.livro if x["evento"] in {"dividendo", "swap"}) == pytest.approx(8 * q)
    r_fut, *_ = executar(atuais, [diaria], especificacoes={"X": spec(modo_contrato="futuro", ajustes=ajustes)})
    assert not [x for x in r_fut.livro if x["evento"] in {"dividendo", "swap"}]
    r_dt, *_ = executar(atuais, [intencao()], especificacoes={"X": spec(modo_contrato="indice_spot", ajustes=ajustes)})
    assert not [x for x in r_dt.livro if x["evento"] == "swap"]


def test_f2_6_indice_sem_historico_recebe_dividendo_pessimista_t3():
    atuais = [
        barra(BASE, h=1005, l=995, c=1000),
        barra(BASE.replace(hour=22), h=1005, l=995, c=1000),
    ]
    i = intencao(campanha=Campanha.DIARIA)
    contrato = spec(
        classe="indice_spot", modo_contrato="indice_spot",
        dividendo_pessimista_usd_por_lote_dia=4.0,
    )
    com_custo, *_ = executar(
        atuais, [i], especificacoes={"X": contrato},
        parametros=params(pessimista=True),
    )
    sem_custo, *_ = executar(
        atuais, [i],
        especificacoes={"X": replace(contrato, dividendo_pessimista_usd_por_lote_dia=0.0)},
        parametros=params(pessimista=True),
    )
    q = aberturas(com_custo)[0]["quantidade"]
    assert com_custo.pnl_usd[-1] - sem_custo.pnl_usd[-1] == pytest.approx(-4.0 * q)


def test_r2_valor_do_ponto_em_moeda_de_cotacao_converte_no_cambio_da_saida():
    entrada = barra(BASE, o=1000, h=1005, l=995, c=1000, cambio=0.006)
    limite = barra(
        BASE.replace(hour=22), o=1100, h=1101, l=1099, c=1100, cambio=0.007
    )
    sp = spec(valor_ponto_moeda_cotacao=100.0)
    r, *_ = executar([entrada, limite], [intencao()], especificacoes={"X": sp})
    a = aberturas(r)[0]
    esperado = (1100 - a["preco_entrada"]) * 100.0 * a["quantidade"] * 0.007
    assert r.pnl_usd[-1] == pytest.approx(esperado)


def test_r3_derrapagem_na_entrada_stop_comum_e_zeragem_pontual():
    sp = spec(slippage_pontos=3)
    entrada = barra(BASE, o=1000, h=1005, l=990, c=1000)
    r, *_ = executar([entrada], [intencao()], especificacoes={"X": sp})
    a = aberturas(r)[0]
    assert a["preco_entrada"] == 1004
    assert fechamentos(r)[0]["preco_saida"] == a["stop"] - 3

    limite = BASE.replace(hour=12, minute=45)
    tranquilas = [
        barra(BASE, o=1000, h=1005, l=995, c=1000),
        barra(limite, o=1010, h=1011, l=1009, c=1010),
    ]
    r2, *_ = executar(
        tranquilas, [intencao(limite=limite)], especificacoes={"X": sp},
        sessoes={("X", BASE.date()): sessao("X", BASE.date(), limite=limite)},
    )
    assert fechamentos(r2)[0]["preco_saida"] == 1010 - 3


def test_r3_pessimista_usa_pior_spread_das_duas_pontas():
    hist = historico()
    hist[-1] = replace(hist[-1], spread=5)
    hist += [
        replace(b, inicio=b.inicio.replace(hour=22), spread=(5 if n == 59 else 1))
        for n, b in enumerate(historico())
    ]
    atuais = [barra(BASE, o=1000, h=1005, l=996, c=1000),
              barra(BASE.replace(hour=22), o=1000, h=1001, l=999, c=1000)]
    r, *_ = executar(atuais, [intencao()], historicos={"X": hist},
                     parametros=params(pessimista=True))
    a, f = aberturas(r)[0], fechamentos(r)[0]
    assert a["spread"] == 5 and a["preco_entrada"] == 1005
    assert f["preco_saida"] == 996  # bid menos alargamento 5-1 na saida


def test_f2_13_zeragem_na_fronteira_e_sem_barra_usa_proxima_com_custo():
    limite = BASE.replace(hour=12, minute=45)
    entrada = barra(BASE, h=1005, l=995, c=1000)
    depois = barra(limite + timedelta(minutes=15), o=1000, h=1005, l=995, c=1000)
    s = {("X", BASE.date()): sessao("X", BASE.date(), limite=limite)}
    r, *_ = executar(
        [entrada, depois], [intencao(limite=limite)],
        especificacoes={"X": spec(slippage_pontos=3)}, sessoes=s,
    )
    f = fechamentos(r)[0]
    assert f["instante"] == limite + timedelta(minutes=15)
    assert f["preco_saida"] == depois.abertura - 3
    tardia = intencao(sinal=limite - timedelta(minutes=5), limite=limite)
    r2, *_ = executar([entrada, depois], [tardia], sessoes=s)
    assert r2.recusas[0].motivo is MotivoRecusa.ZERAGEM


def test_f2_14_filtro_hora_e_causal_a_spread_futuro():
    ruins = historico(spread=30, movimento=100)
    r, *_ = executar([barra(BASE)], [intencao()], historicos={"X": ruins})
    assert r.recusas[0].motivo is MotivoRecusa.FILTRO_HORA
    boas = historico(spread=1, movimento=100)
    atuais = [barra(BASE), barra(BASE.replace(hour=22))]
    r1, *_ = executar(atuais, [intencao()], historicos={"X": boas})
    futuro = barra(BASE + timedelta(days=1), spread=100, h=1100, l=900, c=1000)
    r2, *_ = executar(atuais + [futuro], [intencao()], historicos={"X": boas})
    assert aberturas(r1)[0] == aberturas(r2)[0]


def test_f2_15_igualdade_rejeita_e_dia_seguinte_reabre():
    # Risco calculado em exatamente 45 por ordem; duas ordens chegam a 3% = 90.
    p = params(risco_por_operacao=0.015)
    d2 = BASE + timedelta(days=1)
    atuais = [
        barra(BASE, h=1005, l=995, c=1000),
        barra(BASE, simbolo="Y", h=1005, l=995, c=1000),
        barra(BASE.replace(hour=22), h=1005, l=995, c=1000),
        barra(BASE.replace(hour=22), simbolo="Y", h=1005, l=995, c=1000),
        barra(d2, h=1005, l=995, c=1000),
        barra(d2.replace(hour=22), h=1005, l=995, c=1000),
    ]
    primeira = intencao(stop=9, regra="a")
    segunda = intencao(simbolo="Y", stop=9, regra="b")
    proximo = intencao(sinal=d2, limite=d2.replace(hour=22), stop=9, regra="c")
    r, *_ = executar(atuais, [primeira, segunda, proximo], parametros=p)
    assert any(x.motivo is MotivoRecusa.LIMITE_DIA for x in r.recusas)
    assert any(x["intencao_id"] == proximo.intencao_id for x in aberturas(r))


def test_f2_15_tres_gaps_passam_limite_fecham_e_travam_ate_dia_seguinte():
    p = params(risco_por_operacao=0.0066)
    tempos = [BASE + timedelta(minutes=15 * n) for n in range(7)]
    precos = [1000, 984, 984, 968, 968, 952, 952]
    atuais = [
        barra(t, o=preco, h=preco + 2, l=preco - 5, c=preco)
        for t, preco in zip(tempos, precos, strict=True)
    ]
    d2 = BASE + timedelta(days=1)
    atuais += [barra(d2), barra(d2.replace(hour=22))]
    intents = [
        intencao(sinal=tempos[0], regra="s1"),
        intencao(sinal=tempos[2], regra="s2"),
        intencao(sinal=tempos[4], regra="s3"),
        intencao(sinal=tempos[6], regra="bloqueada"),
        intencao(sinal=d2, limite=d2.replace(hour=22), regra="novo_dia"),
    ]
    hist_duas_horas = historico() + [
        replace(b, inicio=b.inicio + timedelta(hours=1)) for b in historico()
    ]
    r, *_ = executar(atuais, intents, parametros=p, historicos={"X": hist_duas_horas})
    bloqueada = intents[3]
    assert any(x.intencao_id == bloqueada.intencao_id and x.motivo is MotivoRecusa.LIMITE_DIA for x in r.recusas)
    assert any(x["intencao_id"] == intents[4].intencao_id for x in aberturas(r))


def test_f2_8_blocos_trinta_operacoes_no_mesmo_dia_tem_um_rotulo():
    # O motor emite um rotulo por aposta; a Prova conta distintos.
    r, *_ = executar([barra(BASE), barra(BASE.replace(hour=22))], [intencao()])
    assert len(set(r.blocos)) == 1


def test_c1_intencoes_concorrentes_entram_por_peso_depois_id():
    baixa = replace(intencao(regra="baixa"), peso_admissao=1.0)
    alta = replace(intencao(regra="alta"), peso_admissao=3.0)
    r, *_ = executar(
        [barra(BASE), barra(BASE.replace(hour=22))], [baixa, alta],
        parametros=params(risco_por_operacao=0.005),
    )
    assert [x["intencao_id"] for x in aberturas(r)][:1] == [alta.intencao_id]
    assert any(
        x.intencao_id == baixa.intencao_id
        and x.motivo is MotivoRecusa.POSICAO_ABERTA
        for x in r.recusas
    )


def test_netting_recusa_segunda_posicao_na_mesma_direcao():
    primeira = replace(intencao(regra="primeira"), peso_admissao=2.0)
    segunda = replace(intencao(regra="segunda"), peso_admissao=1.0)
    r, *_ = executar(
        [barra(BASE), barra(BASE.replace(hour=22))], [primeira, segunda],
        parametros=params(risco_por_operacao=0.005),
    )
    assert [x["intencao_id"] for x in aberturas(r)] == [primeira.intencao_id]
    assert any(
        x.intencao_id == segunda.intencao_id
        and x.motivo is MotivoRecusa.POSICAO_ABERTA
        for x in r.recusas
    )


def test_netting_recusa_posicao_oposta_no_mesmo_simbolo():
    compra = replace(intencao(regra="compra"), peso_admissao=2.0)
    venda = replace(
        intencao(regra="venda", lado=Lado.VENDA), peso_admissao=1.0
    )
    r, *_ = executar(
        [barra(BASE), barra(BASE.replace(hour=22))], [compra, venda],
        parametros=params(risco_por_operacao=0.005),
    )
    assert [x["intencao_id"] for x in aberturas(r)] == [compra.intencao_id]
    recusa = next(x for x in r.recusas if x.intencao_id == venda.intencao_id)
    assert recusa.motivo is MotivoRecusa.POSICAO_ABERTA
    assert recusa.dados["lado_aberto"] == "compra" and recusa.dados["lado_novo"] == "venda"


def test_netting_simbolo_diferente_continua_aceito():
    x = intencao(regra="x")
    y = intencao(simbolo="Y", regra="y")
    atuais = [
        barra(BASE, simbolo="X"), barra(BASE, simbolo="Y"),
        barra(BASE.replace(hour=22), simbolo="X"),
        barra(BASE.replace(hour=22), simbolo="Y"),
    ]
    r, *_ = executar(atuais, [x, y], parametros=params(risco_por_operacao=0.005))
    assert {a["intencao_id"] for a in aberturas(r)} == {x.intencao_id, y.intencao_id}


@pytest.mark.parametrize("saida", ["stop", "zeragem"])
def test_netting_simbolo_volta_a_aceitar_depois_de_fechar(saida):
    t1 = BASE + timedelta(minutes=15)
    limite_primeira = t1 if saida == "zeragem" else BASE.replace(hour=22)
    primeira = intencao(regra="primeira", limite=limite_primeira)
    segunda = intencao(sinal=t1, regra="segunda")
    b0 = barra(BASE, l=(980 if saida == "stop" else 995))
    atuais = [b0, barra(t1), barra(BASE.replace(hour=22))]
    r, *_ = executar(atuais, [primeira, segunda], parametros=params(risco_por_operacao=0.005))
    assert {a["intencao_id"] for a in aberturas(r)} == {
        primeira.intencao_id, segunda.intencao_id
    }
    assert not any(
        x.intencao_id == segunda.intencao_id
        and x.motivo is MotivoRecusa.POSICAO_ABERTA
        for x in r.recusas
    )


def test_caminho_otimizado_e_referencia_sao_identicos_campo_a_campo():
    t1 = BASE + timedelta(minutes=15)
    atuais = [
        barra(BASE, h=1005, l=995, c=1000),
        barra(t1, o=980, h=985, l=975, c=980),
        barra(BASE.replace(hour=22), o=980, h=985, l=975, c=980),
    ]
    primeira = intencao(regra="primeira")
    concorrente = replace(intencao(regra="concorrente"), peso_admissao=0.5)
    posterior = intencao(sinal=t1, regra="posterior")
    intencoes = [primeira, concorrente, posterior]
    rapido, dados, contratos, calendario = executar(atuais, intencoes)
    lento = simular(
        intencoes, dados, contratos, calendario, params(), _otimizar=False
    )
    assert rapido.livro == lento.livro
    assert rapido.retorno == lento.retorno
    assert rapido.recusas == lento.recusas
    assert rapido.estados == lento.estados
    assert rapido.conferencias_margem == lento.conferencias_margem
    assert rapido == lento


def test_simular_isoladas_compartilha_precalculo_e_preserva_resultados(monkeypatch):
    d2 = BASE + timedelta(days=1)
    atuais = [
        barra(BASE, h=1020, l=995, c=1010),
        barra(BASE.replace(hour=22), o=1010, h=1011, l=1009, c=1010),
        barra(d2, h=1005, l=980, c=990),
        barra(d2.replace(hour=22), o=990, h=991, l=989, c=990),
    ]
    por_chave = {
        "ganha": [intencao(alvo=15, regra="ganha")],
        "perde": [intencao(sinal=d2, limite=d2.replace(hour=22), regra="perde")],
    }
    todos = [i for itens in por_chave.values() for i in itens]
    _, dados, contratos, calendario = executar(atuais, todos)
    esperados = {
        chave: simular(itens, dados, contratos, calendario, params())
        for chave, itens in por_chave.items()
    }
    original = modulo_motor._precalcular_horas
    chamadas = 0

    def contar(*args, **kwargs):
        nonlocal chamadas
        chamadas += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(modulo_motor, "_precalcular_horas", contar)
    obtidos = simular_isoladas(por_chave, dados, contratos, calendario, params())
    assert chamadas == 1
    assert obtidos == esperados


def _mundo_eventos_grande(*, denso=False):
    dados, contratos, calendario = {}, {}, {}
    por_chave = {}
    primeiro = date(2023, 1, 2)
    dias = [primeiro + timedelta(days=n) for n in range(61 if denso else 75)]
    for ativo in range(8):
        simbolo = f"S{ativo}"
        barras = []
        for numero, dia in enumerate(dias):
            calendario[(simbolo, dia)] = sessao(simbolo, dia)
            centro = 10_000 + 100 * ativo + (numero % 9) - 4
            for quarto in range(16):
                instante = datetime.combine(dia, datetime.min.time(), UTC) + timedelta(
                    hours=9, minutes=15 * quarto
                )
                deslocamento = ((quarto + numero + ativo) % 5) - 2
                abertura = centro + deslocamento
                barras.append(BarraMotor(
                    simbolo, instante, "M15", abertura, abertura + 3,
                    abertura - 3, abertura + (1 if quarto % 2 else -1),
                    1 + (numero + quarto) % 3, dia,
                ))
        dados[simbolo] = barras
        contratos[simbolo] = spec(
            simbolo, slippage_pontos=1, margem_por_lote_usd=5.0
        )
        for regra in range(26):
            intencoes = []
            for dia in dias[60:]:
                sinais = (
                    [datetime.combine(dia, datetime.min.time(), UTC)
                     + timedelta(hours=9, minutes=5 * k) for k in range(45)]
                    if denso else
                    [datetime.combine(dia, datetime.min.time(), UTC) + timedelta(hours=10)]
                )
                for sinal in sinais:
                    intencoes.append(Intencao(
                        Campanha.DAY_TRADE_M15, f"R{regra:02d}", simbolo,
                        Lado.COMPRA if regra % 2 == 0 else Lado.VENDA,
                        sinal, 8 + regra % 7,
                        sinal + timedelta(minutes=(-sinal.minute) % 15 + 30) if denso else
                        datetime.combine(dia, datetime.min.time(), UTC)
                        + timedelta(hours=12, minutes=45),
                        None if regra % 3 == 0 else 6 + regra % 5,
                    ))
            por_chave[(simbolo, regra)] = intencoes
    return por_chave, dados, contratos, calendario


@pytest.mark.parametrize("pessimista", [False, True])
@pytest.mark.parametrize("risco", [0.005, 0.01, 0.02])
def test_isoladas_densas_iguais_ao_oraculo_em_208_chaves_3_riscos(
    pessimista, risco,
):
    por_chave, dados, contratos, calendario = _mundo_eventos_grande(denso=True)
    parametros = params(
        pessimista=pessimista, risco_por_operacao=risco,
        capital_constante=True,
    )
    obtidos = simular_isoladas(
        por_chave, dados, contratos, calendario, parametros,
        somente_series=True,
    )
    assert len(obtidos) == 208
    for chave, intencoes in por_chave.items():
        esperado = simular(
            intencoes, dados, contratos, calendario, parametros,
            _otimizar=False,
        )
        atual = obtidos[chave]
        assert (
            atual.datas, atual.pnl_usd, atual.retorno,
            atual.exposicao_comprada, atual.exposicao_vendida,
            atual.blocos,
        ) == (
            esperado.datas, esperado.pnl_usd, esperado.retorno,
            esperado.exposicao_comprada, esperado.exposicao_vendida,
            esperado.blocos,
        )
        assert atual.livro == atual.recusas == atual.estados == ()
        assert atual.conferencias_margem == ()


@pytest.fixture(scope="module")
def mundo_muitas_intencoes():
    inicio = date(2022, 1, 3)
    dias, d = [], inicio
    while len(dias) < 400:
        if d.weekday() < 5:
            dias.append(d)
        d += timedelta(days=1)
    dados, contratos, calendario, todas = {}, {}, {}, []
    for numero, simbolo in enumerate(("A", "B", "C", "D")):
        bs = []
        for k, dia in enumerate(dias):
            t = datetime.combine(dia, datetime.min.time(), UTC) + timedelta(hours=10)
            base = 10_000 + 100 * numero + (k % 7)
            bs.extend([
                barra(t, simbolo=simbolo, o=base, h=base + 12, l=base - 12, c=base, spread=1 + k % 3),
                barra(t + timedelta(minutes=15), simbolo=simbolo, o=base, h=base + 2, l=base - 2,
                      c=base, spread=1 + k % 3),
            ])
            calendario[(simbolo, dia)] = sessao(
                simbolo, dia, limite=t + timedelta(minutes=15)
            )
            if k >= 60:
                for regra in range(160):
                    todas.append(Intencao(
                        Campanha.DAY_TRADE_M15, f"R{regra:03d}", simbolo,
                        Lado.COMPRA if regra % 2 == 0 else Lado.VENDA,
                        t, 8 + regra % 5, t + timedelta(minutes=15),
                        6 + regra % 4, peso_admissao=1.0 + (regra % 3) / 10,
                    ))
        dados[simbolo] = bs
        contratos[simbolo] = spec(simbolo, slippage_pontos=1, margem_por_lote_usd=5.0)
    selecao = [i for i in todas if int(i.regra_id[1:]) % 2 == 0]
    assert len(todas) >= 200_000 and len(selecao) >= 100_000
    return dados, contratos, calendario, todas, selecao


@pytest.mark.parametrize("pessimista", [False, True])
@pytest.mark.parametrize("selecao", [False, True], ids=["V0", "selecao"])
def test_muitas_intencoes_series_iguais_ao_oraculo(
    mundo_muitas_intencoes, pessimista, selecao
):
    dados, contratos, calendario, todas, escolhidas = mundo_muitas_intencoes
    intencoes = escolhidas if selecao else todas
    p = params(
        risco_por_operacao=0.005, pessimista=pessimista,
        capital_constante=True,
    )
    esperado = simular(
        intencoes, dados, contratos, calendario, p,
        somente_series=True, _otimizar=False,
    )
    atual = simular(
        intencoes, dados, contratos, calendario, p,
        somente_series=True,
    )
    assert atual == esperado
    assert atual.livro == atual.recusas == atual.estados == ()
    assert atual.conferencias_margem == ()


@pytest.mark.parametrize("risco", [0.005, 0.01, 0.02])
@pytest.mark.parametrize("selecao", [False, True], ids=["V0", "selecao"])
def test_simular_par_igual_a_duas_referencias_em_4_ativos_400_dias_100k(
    mundo_muitas_intencoes, risco, selecao,
):
    dados, contratos, calendario, todas, escolhidas = mundo_muitas_intencoes
    intencoes = escolhidas if selecao else todas
    normal = params(
        risco_por_operacao=risco, pessimista=False, capital_constante=True,
    )
    pessimista = replace(normal, pessimista=True)
    esperado_normal = simular(
        intencoes, dados, contratos, calendario, normal,
        somente_series=True, _otimizar=False,
    )
    esperado_pessimista = simular(
        intencoes, dados, contratos, calendario, pessimista,
        somente_series=True, _otimizar=False,
    )
    atual_normal, atual_pessimista = simular_par(
        intencoes, dados, contratos, calendario, normal, pessimista,
        somente_series=True,
    )
    assert atual_normal == esperado_normal
    assert atual_pessimista == esperado_pessimista


@pytest.mark.parametrize("pessimista", [False, True])
def test_executor_por_eventos_com_margem_por_taxa_igual_ao_oraculo(pessimista):
    por_chave, dados, contratos, calendario = _mundo_eventos_grande(denso=True)
    por_chave = {k: v for k, v in por_chave.items() if k[0] == "S0" and k[1] < 3}
    contratos = {
        s: replace(v, margem_por_lote_usd=None, taxa_margem_notional=5e-5)
        for s, v in contratos.items()
    }
    p = params(
        pessimista=pessimista, risco_por_operacao=0.005,
        capital_constante=True,
    )
    atuais = simular_isoladas(
        por_chave, dados, contratos, calendario, p, somente_series=True,
    )
    for chave, intencoes in por_chave.items():
        esperado = simular(
            intencoes, dados, contratos, calendario, p, _otimizar=False,
        )
        assert atuais[chave] == replace(
            esperado, livro=(), recusas=(), estados=(), conferencias_margem=(),
        )


def test_executor_por_eventos_taxa_liquida_na_primeira_barra_de_margem():
    barras = historico() + [
        barra(BASE, o=1000, h=1001, l=999, c=1000, cambio=1.0),
        barra(
            BASE + timedelta(minutes=15), o=1000, h=1001, l=999, c=1000,
            cambio=10.0,
        ),
        barra(
            BASE + timedelta(minutes=30), o=1000, h=1001, l=999, c=1000,
            cambio=10.0,
        ),
    ]
    i = intencao(
        lado=Lado.VENDA, stop=100, limite=BASE + timedelta(minutes=45),
    )
    contratos = {
        "X": spec(
            margem_por_lote_usd=None, taxa_margem_notional=0.4,
        )
    }
    calendario = {("X", BASE.date()): sessao("X", BASE.date())}
    p = params(capital_constante=True)
    esperado = simular(
        [i], {"X": barras}, contratos, calendario, p, _otimizar=False,
    )
    atual = simular_isoladas(
        {"r": [i]}, {"X": barras}, contratos, calendario, p,
        somente_series=True,
    )["r"]
    assert fechamentos(esperado)[0]["motivo"] == "reducao_margem"
    assert atual == replace(
        esperado, livro=(), recusas=(), estados=(), conferencias_margem=(),
    )


def test_simular_par_com_margem_por_taxa_igual_a_duas_referencias():
    por_chave, dados, contratos, calendario = _mundo_eventos_grande(denso=True)
    intencoes = [
        i for (simbolo, regra), itens in por_chave.items()
        if simbolo in {"S0", "S1"} and regra < 3 for i in itens
    ]
    contratos = {
        s: replace(v, margem_por_lote_usd=None, taxa_margem_notional=5e-5)
        for s, v in contratos.items()
    }
    normal = params(
        risco_por_operacao=0.005, pessimista=False, capital_constante=True,
    )
    pessimista = replace(normal, pessimista=True)
    esperado = tuple(
        simular(
            intencoes, dados, contratos, calendario, p,
            somente_series=True, _otimizar=False,
        )
        for p in (normal, pessimista)
    )
    assert simular_par(
        intencoes, dados, contratos, calendario, normal, pessimista,
        somente_series=True,
    ) == esperado

