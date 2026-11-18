"""F2-12: estudo de poder do Garimpo em mundos sintéticos (nunca preço real).

Uso: python ferramentas/estudo_poder.py <saida.jsonl> [--ativos 81] [--proc-interno 4] [--shard k/N]
Retomável: mundos já gravados (cenário, semente) são pulados. Cada linha = um mundo × 15 configurações.
Um mundo por vez; dentro dele, gramática e séries isoladas rodam por ativo em --proc-interno processos.
--shard k/N fica com as tarefas de índice i ≡ k (mod N) da lista determinística — para o GitHub Actions.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FORCAS = json.loads((Path(__file__).resolve().parents[1] / "medidas" / "f2_12_forcas_vantagem.json")
                    .read_text(encoding="utf-8"))["forcas"]
CENARIOS = [("nenhuma", 0.0, 0.0)] + [(t, float(sh), FORCAS[t][sh]) for t in ("momento", "reversao", "sazonal")
                                      for sh in ("0.5", "1.0", "2.0")]  # congelado antes de resultado (T12)
# sementes explícitas e imutáveis por (vantagem, Sharpe); F2-9 usa 900_000+
SEMENTE = {("nenhuma", 0.0): 130_000,
           ("momento", 0.5): 131_000, ("momento", 1.0): 132_000, ("momento", 2.0): 133_000,
           ("reversao", 0.5): 134_000, ("reversao", 1.0): 135_000, ("reversao", 2.0): 136_000,
           ("sazonal", 0.5): 137_000, ("sazonal", 1.0): 138_000, ("sazonal", 2.0): 139_000}
# sem vantagem: 119 mundos = Clopper-Pearson 95% ≤ 2,5% com 0 aprovações (titan/certificacao.py)
MUNDOS = {"nenhuma": 119, "momento": 20, "reversao": 20, "sazonal": 20}
N_DIAS, FRACAO = 1240, 0.5  # vantagem em metade dos ativos


def um_mundo(cenario: tuple, k: int, n_ativos: int, proc_interno: int) -> dict:
    from titan import cadeia_garimpo as cg
    from titan.aprendizagem import CONFIGURACOES, RISCOS
    from titan.prova import SHARPE_SUSPEITO
    from titan.sintetico import Vantagem, gerar_mundo
    from titan.tipos_f2 import ParamsPortao

    tipo, sharpe, forca = cenario
    semente = SEMENTE[(tipo, sharpe)] + k
    t0 = time.time()
    v = Vantagem(tipo, forca, FRACAO) if tipo != "nenhuma" else Vantagem()
    w = gerar_mundo(semente, N_DIAS, n_ativos, v)
    lotes, series = cg.preparar(w, RISCOS, processos=proc_interno)
    jan = cg.janelas_padrao(w.dias)
    portao = ParamsPortao(0.025, 0.8, 2, 30, 0.05, 0.2, 499, semente)
    niveis = portao.niveis()
    res = []
    for variante, risco in CONFIGURACOES:
        a = cg.avaliar_configuracao(w, lotes, series[risco], variante, risco, jan, portao)
        ok_resto = a.blocos >= portao.apostas_min and a.p_cofre_neg >= portao.nivel_cofre and a.sharpe <= SHARPE_SUSPEITO
        res.append(dict(variante=variante, risco=risco, p=a.p, blocos=a.blocos, p_cofre_neg=a.p_cofre_neg,
                        sharpe=a.sharpe, media=a.media,
                        aprovado_1=bool(ok_resto and a.p <= niveis[0]), aprovado_2=bool(ok_resto and a.p <= niveis[1])))
    return dict(tipo=tipo, sharpe_plantado=sharpe, forca=forca, semente=semente, n_ativos=n_ativos,
                segundos=round(time.time() - t0, 1), plataforma=platform.platform(), configuracoes=res)


def tarefas_todas() -> list[tuple]:
    out = []
    for k in range(max(MUNDOS.values())):  # intercalado: todos os cenários avançam juntos
        for c in CENARIOS:
            if k < MUNDOS[c[0]]:
                out.append((c, k))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("saida")
    ap.add_argument("--ativos", type=int, default=81)
    ap.add_argument("--proc-interno", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tarefa", type=int, default=None, help="roda só a tarefa de índice i (um mundo)")
    ap.add_argument("--listar", action="store_true", help="imprime o número de tarefas e sai")
    a = ap.parse_args()
    if a.listar:
        print(len(tarefas_todas()))
        return
    k_shard, n_shard = (int(x) for x in a.shard.split("/"))
    saida = Path(a.saida)
    feitos = set()
    if saida.exists():
        for linha in saida.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                d = json.loads(linha)
                feitos.add((d["tipo"], d["sharpe_plantado"], d["semente"]))
    todas = tarefas_todas()
    if a.tarefa is not None:
        minhas = [todas[a.tarefa]]
    else:
        minhas = [t for i, t in enumerate(todas) if i % n_shard == k_shard]
    with saida.open("a", encoding="utf-8") as f:
        for c, k in minhas:
            if (c[0], c[1], SEMENTE[(c[0], c[1])] + k) in feitos:
                continue
            d = um_mundo(c, k, a.ativos, a.proc_interno)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush()
            print(d["tipo"], d["sharpe_plantado"], d["semente"], d["segundos"], "s", flush=True)


if __name__ == "__main__":
    main()
