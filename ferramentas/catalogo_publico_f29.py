"""Catálogo mínimo e público usado exclusivamente pelos mundos sintéticos F2-9."""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.motor import EspecificacaoAtivo
from titan.tipos import Sessao


RAIZ = Path(__file__).resolve().parents[1]
ORIGEM_PADRAO = RAIZ / "medidas" / "catalogo_f2.json"
DESTINO_PADRAO = RAIZ / "medidas" / "catalogo_f2_publico.json"
VERSAO = 1
SPEC_CAMPOS = (
    "classe", "valor_ponto_moeda_cotacao", "lote_minimo", "passo_lote",
    "margem_por_lote_usd", "stop_out_nivel_margem", "taxa_margem_notional",
    "slippage_pontos", "stop_minimo_pontos", "modo_contrato",
)
SESSAO_CAMPOS = (
    "data_servidor", "abertura_utc", "fechamento_utc", "limite_zeragem_utc",
    "virada_utc", "offset_servidor_min", "feriado",
)


def gerar(origem: Path = ORIGEM_PADRAO, destino: Path = DESTINO_PADRAO) -> Path:
    bruto = json.loads(Path(origem).read_text(encoding="utf-8"))
    if bruto.get("sem_preco") is not True or not isinstance(bruto.get("simbolos"), dict):
        raise ValueError("catálogo F2 de origem inválido")
    simbolos = {}
    for simbolo, registro in sorted(bruto["simbolos"].items()):
        e = registro["especificacao"]
        faltam_e = set(SPEC_CAMPOS) - set(e)
        if faltam_e:
            raise ValueError(f"especificação incompleta de {simbolo}: {sorted(faltam_e)}")
        sessoes = []
        for s in registro["sessoes"]:
            faltam_s = set(SESSAO_CAMPOS) - set(s)
            if faltam_s:
                raise ValueError(f"sessão incompleta de {simbolo}: {sorted(faltam_s)}")
            sessoes.append({k: s[k] for k in SESSAO_CAMPOS})
        simbolos[simbolo] = {
            "especificacao": {k: e[k] for k in SPEC_CAMPOS},
            "sessoes": sessoes,
        }
    saida = {
        "versao": VERSAO,
        "sem_preco": True,
        "finalidade": "mundos sintéticos F2-9",
        "historia_sinal_embaralhado": "fora: exigiria preço real",
        "simbolos": simbolos,
    }
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    tmp.write_text(json.dumps(saida, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
                   encoding="utf-8")
    tmp.replace(destino)
    return destino


def carregar(caminho: Path = DESTINO_PADRAO):
    bruto = json.loads(Path(caminho).read_text(encoding="utf-8"))
    if bruto.get("versao") != VERSAO or bruto.get("sem_preco") is not True:
        raise ValueError("catálogo público F2-9 inválido")
    sessoes, specs = {}, {}
    for simbolo, registro in bruto["simbolos"].items():
        e = registro["especificacao"]
        if set(e) != set(SPEC_CAMPOS):
            raise ValueError(f"campos de especificação divergentes em {simbolo}")
        specs[simbolo] = EspecificacaoAtivo(simbolo=simbolo, **e)
        por_dia = {}
        for item in registro["sessoes"]:
            if set(item) != set(SESSAO_CAMPOS):
                raise ValueError(f"campos de sessão divergentes em {simbolo}")
            d = date.fromisoformat(item["data_servidor"])
            instante = lambda nome: None if item[nome] is None else datetime.fromisoformat(item[nome])
            por_dia[d] = Sessao(
                simbolo, d, instante("abertura_utc"), instante("fechamento_utc"),
                instante("limite_zeragem_utc"), datetime.fromisoformat(item["virada_utc"]),
                int(item["offset_servidor_min"]), bool(item["feriado"]),
                "catalogo_publico_f29",
            )
        sessoes[simbolo] = por_dia
    return sessoes, specs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--origem", type=Path, default=ORIGEM_PADRAO)
    p.add_argument("--saida", type=Path, default=DESTINO_PADRAO)
    a = p.parse_args()
    print(gerar(a.origem, a.saida))


if __name__ == "__main__":
    main()
