"""F2-2 na cadeia inteira do Garimpo: gramática → séries isoladas → aprendizagem → aplicação → motor.

Teste do plano: "o resultado não muda ao alterar dado anterior ao aquecimento; cortar a série em 200 datas
sorteadas: as decisões até cada data são idênticas às da série inteira."

Para cada corte t e cada modo (apagar, alterar, acrescentar o que vem depois de t, em todos os ativos):
  (a) as intenções aplicadas (id e peso) com sinal <= t são idênticas;
  (b) o retorno diário da série normal é idêntico em todo dia de servidor < dia de t.
E uma mutação SÓ anterior ao início declarado (dado antes de dias[0]) não muda nada.

Uso: python ferramentas/medir_causalidade_cadeia.py [n_cortes] [saida.json]
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan import cadeia_garimpo as cg  # noqa: E402
from titan.aprendizagem import agenda, aplicar, dia_servidor  # noqa: E402
from titan.calendario import meia_noite_servidor_utc  # noqa: E402
from titan.causalidade import MODOS  # noqa: E402
from titan.sintetico import M15, Mundo, Vantagem, gerar_mundo  # noqa: E402
from titan.visao import Barras, ns  # noqa: E402

VARIANTE, RISCO = "V1", 0.01


def recortar(w: Mundo) -> Mundo:
    """O sistema só vê dado a partir do início declarado (dias[0])."""
    t0 = ns(meia_noite_servidor_utc(w.dias[0]))
    m15 = {}
    for s, b in w.m15.items():
        k = int(np.searchsorted(b.ts, t0))
        m15[s] = b.fatia(k, len(b))
    return Mundo(w.dias, m15, w.sessoes, w.specs, w.com_vantagem, dict(w.lado_plantado))


def cadeia(w: Mundo):
    w = recortar(w)
    its = cg.intencoes(w)
    ser = cg.series_isoladas(w, its, RISCO)
    per = agenda(VARIANTE, w.dias, ser, reamostragens=99, semente=1)
    aplicadas = aplicar([i for s in sorted(its) for i in its[s]], per)
    r = cg._rodar(aplicadas, w, cg.params_motor(RISCO, False), enxuto=True)
    return aplicadas, dict(zip(r.datas, r.retorno))


def mutar(w: Mundo, corte, modo: str, rng: random.Random) -> Mundo:
    m15 = {s: MODOS[modo](b, corte, rng) for s, b in w.m15.items()}
    return Mundo(w.dias, m15, w.sessoes, w.specs, w.com_vantagem, dict(w.lado_plantado))


def mutar_antes(w: Mundo, rng: random.Random) -> Mundo:
    """Põe 30 dias de barras inventadas ANTES de dias[0]."""
    m15 = {}
    for s, b in w.m15.items():
        n = 30 * 96
        ts = b.ts[0] - M15 * np.arange(n, 0, -1)
        c = b.c[0] + np.cumsum(np.array([rng.randint(-500, 500) for _ in range(n)], dtype=np.int64))
        o = np.r_[c[0], c[:-1]]
        pre = Barras("M15", ts, o, np.maximum(o, c) + 5, np.minimum(o, c) - 5, c, np.full(n, 8, np.int64))
        m15[s] = Barras("M15", np.r_[pre.ts, b.ts], np.r_[pre.o, b.o], np.r_[pre.h, b.h], np.r_[pre.l, b.l],
                        np.r_[pre.c, b.c], np.r_[pre.spread, b.spread])
    return Mundo(w.dias, m15, w.sessoes, w.specs, w.com_vantagem, dict(w.lado_plantado))


def conferir(n_cortes: int, semente: int = 8, n_dias: int = 420, n_ativos: int = 2) -> dict:
    t0 = time.time()
    w = gerar_mundo(77_000, n_dias, n_ativos, Vantagem("momento", 0.1, 0.5))
    cheia_its, cheia_ret = cadeia(w)
    rng = random.Random(semente)
    tempos = sorted({i.barra_sinal_fim for i in cheia_its})
    cortes = sorted(rng.choice(tempos) for _ in range(n_cortes))
    falhas = []
    ids = lambda xs, t: sorted((i.intencao_id, i.peso_admissao) for i in xs if i.barra_sinal_fim <= t)
    for corte in cortes:
        dia_corte = dia_servidor(corte)
        esperado_its = ids(cheia_its, corte)
        esperado_ret = {d: v for d, v in cheia_ret.items() if d < dia_corte}
        for modo in MODOS:
            its_m, ret_m = cadeia(mutar(w, corte, modo, rng))
            got_ret = {d: v for d, v in ret_m.items() if d < dia_corte}
            if ids(its_m, corte) != esperado_its or got_ret != esperado_ret:
                falhas.append(dict(corte=corte.isoformat(), modo=modo,
                                   intencoes_iguais=ids(its_m, corte) == esperado_its,
                                   retornos_iguais=got_ret == esperado_ret))
    its_a, ret_a = cadeia(mutar_antes(w, rng))
    antes_ok = ids(its_a, max(tempos)) == ids(cheia_its, max(tempos)) and ret_a == cheia_ret
    return dict(n_cortes=n_cortes, modos=list(MODOS), mundo=dict(semente=77_000, n_dias=n_dias, n_ativos=n_ativos),
                variante=VARIANTE, risco=RISCO, falhas=falhas, antes_do_inicio_inalterado=antes_ok,
                intencoes_cheia=len(cheia_its), segundos=round(time.time() - t0, 1))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = conferir(n)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "falhas"}, ensure_ascii=False), "falhas:", len(out["falhas"]))
    print(out["falhas"][:5])
