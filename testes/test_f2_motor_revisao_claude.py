"""Sondas da revisão T12 de Claude ao motor (mesa t12). Devem passar depois do reparo do Codex."""
from datetime import timedelta

from test_f2_motor import BASE, aberturas, barra, executar, intencao, params, spec


def _gap_e_dia_seguinte():
    d2 = BASE + timedelta(days=1)
    atuais = [
        barra(BASE, o=1000, h=1001, l=999, c=1000),
        barra(BASE + timedelta(minutes=15), o=850, h=851, l=849, c=850),  # gap de 150 pontos abaixo do stop
        barra(BASE.replace(hour=22), o=850, h=851, l=849, c=850),
        barra(d2, o=850, h=851, l=849, c=850),
        barra(d2.replace(hour=22), o=850, h=851, l=849, c=850),
    ]
    intents = [intencao(regra="perde"), intencao(sinal=d2, limite=d2.replace(hour=22), regra="seguinte")]
    return atuais, intents


def test_r6_serie_de_prova_nao_revoga_nem_para_de_operar():
    atuais, intents = _gap_e_dia_seguinte()
    r, *_ = executar(atuais, intents)
    assert len(aberturas(r)) == 2, [x.motivo for x in r.recusas]


def test_r6_teto_de_risco_aberto_nao_cresce_com_lucro_passado():
    # Lucro enorme no dia 1 não pode admitir mais risco aberto no dia 2 a capital constante.
    d2 = BASE + timedelta(days=1)
    atuais = [barra(BASE, o=1000, h=9000, l=999, c=9000), barra(BASE.replace(hour=22), o=9000, h=9001, l=8999, c=9000)]
    for k in range(8):
        atuais.append(barra(d2 + timedelta(minutes=0), simbolo=f"S{k}", o=1000, h=1001, l=999, c=1000))
        atuais.append(barra(d2.replace(hour=22), simbolo=f"S{k}", o=1000, h=1001, l=999, c=1000))
    intents = [intencao(alvo=7000, regra="ganha")]
    intents += [intencao(simbolo=f"S{k}", sinal=d2 - timedelta(minutes=15), limite=d2.replace(hour=22), regra=f"s{k}")
                for k in range(8)]
    contratos = {i.simbolo: spec(i.simbolo) for i in intents}
    r_base, *_ = executar([b for b in atuais if b.simbolo != "X"], intents[1:], especificacoes=contratos,
                          parametros=params(limite_perda_dia=1.0))
    r, *_ = executar(atuais, intents, especificacoes=contratos, parametros=params(limite_perda_dia=1.0))
    dia2 = lambda res: [a for a in aberturas(res) if a["intencao_id"] != intents[0].intencao_id]  # noqa: E731
    assert len(dia2(r)) == len(dia2(r_base))
