# Contrato das intenções compactas — campanha 1

Implementação de referência: `titan/intencoes_compactas.py`.

Este contrato substitui a materialização antecipada de um objeto `Intencao` por sinal. Ele não altera a
identidade, a ordem, a deduplicação nem nenhuma regra econômica. A gramática produz um lote por ativo; a
aprendizagem devolve lotes com linhas filtradas e pesos atualizados; o motor materializa `Intencao` somente
quando uma linha admitida ou registrada exige o objeto.

## Interface pública

```python
@dataclass(frozen=True, slots=True)
class LoteIntencoesCompactas:
    campanha: Campanha                 # escalar; na campanha 1, DAY_TRADE_M15
    simbolo: str                       # escalar; exatamente um ativo por lote
    regra_ids: tuple[str, ...]         # tabela local imutável e sem repetição
    regra: np.ndarray                  # uint16, índice em regra_ids
    lado: np.ndarray                   # int8: +1 compra, -1 venda
    sinal_fim_ns: np.ndarray           # int64 UTC desde Unix epoch
    stop_pontos: np.ndarray            # int64, estritamente positivo
    hora_limite_ns: np.ndarray         # int64 UTC desde Unix epoch
    alvo_pontos: np.ndarray            # int64: 0 = None; positivo = alvo
    peso_admissao: np.ndarray          # float64; >= 0, +inf permitido, NaN proibido

    def __len__(self) -> int: ...
    def regra_id(self, linha: int) -> str: ...
    def intencao_id(self, linha: int) -> str: ...
    def materializar(self, linha: int) -> Intencao: ...
    def selecionar(self, linhas: np.ndarray) -> "LoteIntencoesCompactas": ...
```

Todos os arrays são NumPy, unidimensionais, C-contíguos, somente leitura e têm o mesmo comprimento. A classe
recusa conversão implícita de dtype: o produtor entrega exatamente os tipos acima. `regra_ids` usa a ordem de
`regras_do_ativo(simbolo)`; códigos não usados podem permanecer na tabela. O limite de 65.536 regras por ativo
é parte do contrato.

`selecionar` recebe índices `int64` crescentes, sem repetição, e devolve um lote canônico independente. Para
evitar uma cópia quando o motor apenas agrupa séries isoladas, ele pode manter internamente `(lote, indices)`;
isso não é uma segunda interface para os produtores.

## Validação e ordem canônica

Além dos invariantes de `Intencao`, os dois instantes devem ser múltiplos de 1.000 ns (representáveis sem perda
por `datetime`) e `hora_limite_ns` deve ser fronteira UTC de 15 minutos. O construtor exige linhas em ordem
lexicográfica por `(sinal_fim_ns, intencao_id)` e identidades únicas. O produtor pode chamar:

```python
LoteIntencoesCompactas.criar(
    campanha, simbolo, regra_ids, regra, lado, sinal_fim_ns,
    stop_pontos, hora_limite_ns, alvo_pontos, peso_admissao,
    canonicalizar=True,
)
```

`canonicalizar=True` calcula temporariamente os 10 bytes do hash por linha, aplica a mesma deduplicação
`setdefault` do gerador atual e ordena pelo mesmo desempate; os hashes não ficam armazenados. Uma colisão dos
20 hexadecimais mantém a primeira linha, exatamente como hoje. O construtor direto, usado pelo caminho rápido,
apenas valida que o lote já está canônico.

## Identidade idêntica à atual

Para a linha `k`, forma-se sem JSON e sem normalização adicional:

```python
chave = "|".join((
    lote.campanha.value,
    lote.regra_ids[int(lote.regra[k])],
    lote.simbolo,
    "compra" if lote.lado[k] == 1 else "venda",
    datetime_utc_exata(lote.sinal_fim_ns[k]).isoformat(),
    str(int(lote.stop_pontos[k])),
    "None" if lote.alvo_pontos[k] == 0 else str(int(lote.alvo_pontos[k])),
    datetime_utc_exata(lote.hora_limite_ns[k]).isoformat(),
))
id = hashlib.sha256(chave.encode()).hexdigest()[:20]
```

`datetime_utc_exata(ns)` é `EPOCH_UTC + timedelta(microseconds=int(ns)//1000)`, nunca conversão por ponto
flutuante; seu `isoformat()` conserva `+00:00` e a mesma emissão de microssegundos do objeto atual.
`peso_admissao` continua fora da identidade. `materializar(k)` passa o ID calculado explicitamente ao
construtor de `Intencao`; a regressão exige igualdade de todos os campos e do ID com o caminho antigo.

## Portas da gramática, aprendizagem e motor

```python
# gramática
gerar_compacto(simbolo: str, m15: Barras,
                sessoes: Mapping[date, Sessao]) -> LoteIntencoesCompactas

# aprendizagem; preserva a ordem relativa, remove peso zero e troca só peso_admissao
aplicar_compacto(lotes: Mapping[str, LoteIntencoesCompactas],
                 periodos: Sequence[Periodo]) -> dict[str, LoteIntencoesCompactas]

# motor; durante a transição, as formas antigas com Intencao continuam aceitas
simular_isoladas(intencoes_por_chave: Mapping[Chave, LoteOuSelecao], ...) -> dict[Chave, ResultadoMotor]
simular_par(intencoes: Iterable[LoteIntencoesCompactas] | Iterable[Intencao], ...)
```

A aprendizagem localiza o período pelo dia de servidor derivado de `sinal_fim_ns`, consulta a chave
`(regra_id, simbolo)`, elimina peso zero e escreve `float64` com o peso do período. Nenhum outro array muda.
O motor compara prioridades por `(-peso_admissao, intencao_id)` e mantém NETTING e toda a sequência atual de
recusas. A aceitação da interface depende de uma regressão que compare o caminho compacto ao atual, campo a
campo, incluindo IDs, nas saídas da gramática, da aprendizagem, das isoladas e do par.
