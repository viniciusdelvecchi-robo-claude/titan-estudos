# Protocolo da prova — campanha 1 (pré-registro 8, F2-7)

Rascunho em construção. Parte A (estatística e portão) completa; partes B (gramática), C (aprendizagem do
Garimpo) e D (saída do ato) a seguir. Congela-se inteiro, com hash entregue ao proprietário (F2-11), antes de
qualquer resultado. Quem reimplementar pela leitura deste texto deve chegar aos mesmos números, sem ler
`titan/prova.py`.

## A. Estatística e portão

**A1. Entrada.** Para cada candidato (Núcleo e Garimpo), o motor entrega duas séries diárias de retorno
`r_t = pnl_t / 3000` (dia de servidor), a normal e a pessimista (custo do lado ruim da lista T3), e a lista
de rótulos de bloco das apostas. O rótulo de bloco de uma aposta começa pela data ISO (`AAAA-MM-DD`) do dia
de servidor da entrada; todas as apostas de um mesmo dia formam um bloco, em qualquer ativo. Conta-se o
número de datas distintas (os dez primeiros caracteres do rótulo), não de rótulos.

**A2. Trechos.** `janela_inicio` ≤ dia < `cofre_inicio` é a prova; dia ≥ `cofre_inicio` é o Cofre. Dias antes
de `janela_inicio` são aquecimento e não entram em nenhuma estatística.

**A3. Hipótese nula.** A média de `r_t` na prova é ≤ 0. Alternativa: > 0. Unilateral.

**A4. Newey-West.** `n` = dias da prova; `L = floor(4·(n/100)^(2/9))`. Com `d_t = r_t − média`,
`γ_l = (1/n)·Σ_{t>l} d_t·d_{t−l}`, `S = γ_0 + 2·Σ_{l=1..L} (1 − l/(L+1))·γ_l`, `var = max(S,0)/n`.
`t = média/√var`; `p = 1 − Φ(t)` (normal padrão). Se `var = 0`: `p = 0` se média > 0, senão 1.

**A5. Bloco de Politis e White** (2004, com a correção de Patton, Politis e White, 2009), sobre a série da
prova: `K = max(5, ceil(√log10 n))`, `m_max = ceil(√n) + K`, `b_max = ceil(min(3√n, n/3))`,
`c = 2·√(log10 n / n)`. `ρ_k = γ_k/γ_0` para k = 1..m_max+K. `m̂` = menor m em 1..m_max tal que
`|ρ_{m+1}|, …, |ρ_{m+K}|` são todos < c (se nenhum, `m̂ = m_max`). `M = min(2m̂, m_max)`.
Trapézio plano `λ(s) = 1` se |s| ≤ ½; `2(1−|s|)` se ½ < |s| ≤ 1; 0 fora. Com `R_k = γ_|k|`:
`G = Σ_{k=−M..M} λ(k/M)·|k|·R_k`, `g = Σ_{k=−M..M} λ(k/M)·R_k`. Se `g ≤ 0`, bloco = 1.
Senão `b = (G²/g²)^(1/3)·n^(1/3)`, limitado a [1, b_max]. Se `γ_0 = 0`, bloco = 1.

**A6. Bootstrap estacionário estudentizado** (Politis e Romano, 1994). `B` reamostragens, gerador
`numpy.random.default_rng(semente)` (semente da série normal = `semente`; da pessimista = `semente+1`).
Para cada reamostragem, nesta ordem de sorteio: matriz `U` B×n de `rng.random`, depois matriz `I` B×n de
`rng.integers(0, n)`. Índice da posição 0 = `I[·,0]`; posição j>0 = `I[·,j]` se `U[·,j] < 1/b`, senão
`(índice anterior + 1) mod n`. Na amostra `x*`, `t* = (média* − média)/√var*`, com `var*` pela fórmula A4
e o mesmo L da série original (t* = 0 se var* = 0). `p = (1 + #{t* ≥ t})/(B + 1)`, t de A4. As reamostragens
podem ser processadas em lotes; o sorteio é feito lote a lote na ordem acima, com lotes de
`min(B, floor(2.000.000/n))` linhas.

**A7. p do candidato.** O maior dos quatro: Newey-West e bootstrap, nas séries normal e pessimista.

**A8. Ordem de Holm.** Candidatos em ordem crescente de p (empate: nome). O número de candidatos tem de
ser igual a `passos_holm`. Níveis do passo i (0, 1, …): `folga·fatia/(passos − i)`. Campanha 1: fatia 2,5%,
folga 0,8, dois passos → 1,0% e 2,0%. O passo i+1 só recebe nível se o candidato do passo i foi aprovado
(A9 inteiro); senão o candidato fica sem nível, com motivo `holm_parou`.

**A9. Aprovação.** Aprovado se e só se nenhum motivo abaixo:
- `p_acima_do_nivel`: p > nível do passo;
- `blocos_insuficientes`: blocos distintos com data na prova < `apostas_min` (30);
- `veto_do_cofre`: no Cofre da série normal (≥ 2 dias), `Φ(t_cofre) < nivel_cofre` (5%), t_cofre por A4;
- `sentinela_revogaria`: fração das partidas mensais do sistema completo que chegam a revogado em 3 anos
  > `revogacao_max` (20%); sem partidas, fração = 1;
- `retido_T13`: Sharpe anual da prova `média/desvio(ddof=1)·√260,3` > 3,0 — fica retido até a auditoria de
  vazamento.

**A10. Diagnóstico de deriva (emenda C2), não decide.** Por candidato: média da prova, médias das exposições
comprada e vendida da série normal, Sharpe anual.

**A11b. Núcleo.** Momento intradiário (Gao, Han, Li e Zhou, 2018) nos índices à vista de `titan/regras.py`
(`PREGOES`: quatro americanos, sete europeus, UK100; Ásia e Brasil fora). Sinal = lado do retorno do
fechamento do pregão à vista anterior (até 5 dias corridos para trás) até 30 min depois da abertura; zero ou
barra ausente = dia sem sinal. Posição na última meia hora: entrada no fechamento−30 min, saída no
fechamento; onde o CFD para junto com o pregão (americanos), a janela recua 15 min (emenda C7). Stop =
`ceil(3 × mediana |movimento da janela|)` nos 60 pregões anteriores com a janela completa; menos de 20 =
dia sem sinal.

**A11. Parâmetros sem padrão.** fatia, folga, passos, apostas_min, nivel_cofre, revogacao_max, B e semente
são obrigatórios; faltando um, a prova não roda. O capital da série é US$ 3.000 (F2-1, F2-1b).

## B. Gramática do Garimpo — campanha 1 (acordada com o Codex em 16/11/2026, mesa t12)

**B0. Universo.** Câmbio, índices à vista, metais e energia com M15 apto pelo Validador no dia (F2-3c);
contratos a termo se o Validador os liberar (F1-6). Sem ações, sem cripto. Entra e sai por aptidão do dia.
Universo e aptidão são pré-condição do chamador: `gerar` recebe só ativo e dias já admitidos (T12 do Codex).

**B0b. Referências do dia, por ativo.** A = abertura da sessão do calendário (F1-13). L = mín(limite de
zeragem da sessão, 22:45 do servidor): fora da hora 23 do servidor, que tem 66–134% de spread/movimento no
câmbio. Toda posição sai na abertura da barra que começa em L (emenda C7). Índices com pregão à vista
(A11b) têm também a referência "à vista" (abertura e fechamento da bolsa); as famílias 1 e 2 rodam nas duas.

**B0c. Stop comum.** Se a família não diz outro: `ceil(3 × ATR20)` do tempo gráfico da regra, ATR20 =
média simples do true range das 20 barras com fim ≤ instante do sinal (T12).

**B1. Momento intradiário.** Formação: retorno do fechamento da sessão anterior até A+f, f ∈ {30, 60, 120} min.
Manutenção: entra em L−h, sai em L, h ∈ {30, 60} min. Lado = sinal da formação; zero = sem sinal.
6 regras por referência.

**B2. Rompimento da faixa de abertura (Crabel, 1990).** Faixa = máxima e mínima das barras M15 em [A, A+f),
f ∈ {30, 60, 120} min. Primeira barra M15 que fecha fora da faixa, até L−60 min, dá a entrada na abertura da
seguinte, no lado do rompimento; uma por dia. Stop = k × largura da faixa, k ∈ {0,5; 1}. Sai em L.
6 regras por referência.

**B3. Sazonalidade intradiária (Heston, Korajczyk e Sadka, 2010).** Fatias de 30 min do dia de servidor entre
A e L−30. No início de cada fatia, média do retorno da mesma fatia nos D pregões anteriores, D ∈ {20, 60, 120};
opera se |t| > q, q ∈ {1, 2} (t = média/erro-padrão simples; q = 0, "sempre opera", foi retirado em 17/11 por
acordo Claude+Codex — degenerescência e custo computacional, sem olhar resultado por regra, antes do congelamento), no lado da média; sai no fim da fatia. Média e desvio amostral
por soma exatamente arredondada (`math.fsum`): m = fsum(x)/D, s = √(fsum((x−m)²)/(D−1)); com isso o sinal de m
perto de zero não depende da ordem de soma nem da biblioteca (T12 do Codex na F2-7). m = 0 exato = sem sinal.
Retorno da fatia [A+δ, A+δ+30) de um dia = c₁/c₀ − 1, retorno simples, com c₀ = fechamento da barra M15 que
TERMINA exatamente em A+δ e c₁ = fechamento da barra que termina exatamente em A+δ+30 min; faltando qualquer
uma das duas (por exemplo, c₀ da primeira fatia da segunda-feira, cuja barra anterior é a de sexta), a fatia é
incompleta naquele dia e não entra no histórico. Os D pregões são os D últimos dias com a fatia completa, antes
do dia corrente (o retorno do dia só entra depois de decidido o dia).
Erro-padrão zero ou menos de D pregões com a fatia completa = sem sinal.
6 regras.

**B4. Reversão curta (Jegadeesh, 1990; Lehmann, 1990).** Tempo gráfico ∈ {M15, H1}. Se o retorno da última
barra passa de k × σ (σ = desvio amostral dos retornos de todas as barras completas do mesmo tempo gráfico
nos 20 pregões anteriores, contando só retornos entre barras adjacentes e cuja barra de origem está dentro
da janela; σ = 0 = sem sinal; σ pela mesma soma exata da B3; o retorno do sinal também exige barras adjacentes), k ∈ {2, 3},
entra contra; segura H ∈ {1, 4} barras, saindo antes se chegar L. 8 regras.

**B5. Tendência e cruzamento de médias (Brock, Lakonishok e LeBaron, 1992).** Tempo gráfico ∈ {M15, H1};
médias simples (curta, longa) ∈ {(5, 20), (10, 50), (20, 100)} barras. Entra no primeiro cruzamento do dia,
no lado da curta; sai em L ou no stop (B0c); só entra entre A e L−60 min. Simplificação aceita no T12 do
Codex, congelada antes de qualquer preço. Um estado por regra×ativo; o cruzamento
é observado no fechamento da barra e executado na abertura da seguinte (F2-3). 6 regras.

**B6. Núcleo literal.** Nos índices de PREGOES, a própria regra do Núcleo (A11b), mesma classe e mesmo regra_id (`nucleo_momento_intradiario`), avaliada uma vez por data de sessão; assim o Garimpo contém o Núcleo de fato (D3). 1 regra.

Total: 32 regras por ativo, 45 nos índices com pregão à vista. Todas de números redondos e fixadas aqui,
antes de qualquer resultado.

## C. Aprendizagem do Garimpo (acordada; a variante sai do estudo de poder F2-12)

**C0.** Reajuste anual, efetivo no primeiro pregão em ou após 1º de janeiro (data do servidor); antes do primeiro reajuste com 12 meses de
passado, pesos iguais. Cada reajuste usa só as séries regra×ativo produzidas pelo motor até a véspera.

**C1. O que o peso faz.** Regra com peso 0 não emite. Entre as que emitem, quando o orçamento de risco não
comporta todas as intenções da mesma barra, a ordem de admissão é peso decrescente, depois `intencao_id`
(`Intencao.peso_admissao`, fora do hash da identidade).

**C2. Variantes declaradas** (concorrem no F2-12; vence a de maior poder médio entre as que respeitam o nível):
- V0: pesos iguais, sem aprendizagem.
- V1: peso = max(0, t de Newey-West da série regra×ativo, passado inteiro).
- V2: as N regra×ativo de maior t, N ∈ {20, 50}, pesos iguais; empate por (regra, ativo) em ordem
  alfabética; com menos de N elegíveis, entram todos.
- V3: filtro em passos de Romano e Wolf (2005) / Hsu, Hsu e Kuan (2010) a 10% sobre o passado inteiro,
  com bootstrap estacionário de A6; sobreviventes com peso igual; nenhum sobrevivente = não opera naquele ano.

**C2b. Precisões (titan/aprendizagem.py, 16/11).** Séries regra×ativo alinhadas na grade comum de dias de
servidor, 0 nos dias sem posição. Elegível = ao menos dois dias com retorno ≠ 0 no passado; inelegível tem
peso 0 em V1–V3. "12 meses de passado" = o ano civil anterior inteiro na grade (a grade começa antes dele ou
na primeira semana de janeiro dele). t de V1/V2 = t de A4. V3: índices do bootstrap comuns a todas as séries
(sorteio de A6, lote a lote), bloco = mediana dos blocos de A5 das séries elegíveis, semente = semente + ano
do reajuste; em cada passo, rejeita as hipóteses com t > quantil 90% (método "higher") do máximo de t* sobre
as ainda não rejeitadas, até não haver nova rejeição. Configurações do F2-12: {V0, V1, V2-20, V2-50, V3} ×
{0,5; 1; 2}% = 15.

**C3. Risco por operação.** Na campanha 1, 0,5%, 1% ou 2% do capital (2% é teto; respeita o lote mínimo).
Concorre junto com a aprendizagem: o estudo de poder (F2-12) compara as configurações conjuntas
variante × risco, e o Núcleo usa o mesmo risco que vencer. Motivo: com 2% e o limite do dia de 3% com a
trava C6, cabe uma posição aberta por dia (revisão T12, R5).


## D. Ato de confirmação e saída (plano v29, seção 3 e F3-2; rascunho do Claude, 17/11, para T12 do Codex)

**D0. Itens de borda.** (a) Ativo que entra no universo apto passa a emitir no primeiro pregão em que está apto
(F2-3c); o que sai deixa de emitir, e a posição aberta segue até stop ou L. (b) Símbolo sem dado no Cofre: a
posição aberta é dada como encerrada no stop (pior caso), e o dia entra na série. (c) Aquecimento: dias antes de
janela_inicio (A2) — o primeiro reajuste da aprendizagem (C0) — só aquecem. (d) Calendário de pregões,
zeragem (B0b, emenda C7), filtro de custo por hora (F2-14) e limite de perda do dia com trava (emenda C6) são
os do motor congelado no manifesto.

**D1. Um ato só.** Com tudo congelado e o hash entregue ao proprietário (F2-11), o executor baixa o Cofre, roda
o Validador com as regras congeladas e roda Núcleo e Garimpo de uma vez sobre a janela inteira, do fim do
aquecimento até a data do ato. Nada é mostrado antes de a saída inteira estar gravada; o executor grava as
métricas no fim, de uma vez.

**D2. Veredito.** Portão da parte A (A8 Holm, A9 motivos), com as partidas da sentinela do sistema completo
(F2-10) por candidato. Um candidato é aprovado só se nenhum motivo de A9 se aplicar.

**D3. Precedência.** Se os dois passarem, opera-se o Garimpo, que contém a regra do Núcleo na gramática
(B6, literal). Se só um passar, opera-se esse. Se nenhum passar, nada opera e a fatia está gasta.

**D4. Saída pré-definida, e só ela** (diagnóstico não aprova nada):
1. veredito de cada candidato, ordem de Holm, nível aplicado e motivos;
2. os quatro p-valores (A4 e A6, normal e pessimista) e o número de blocos;
3. o resultado do Cofre em separado (média, t, p unilateral negativo) e o resultado por terços da janela de prova;
4. o sistema completo partindo de cada mês: chance de revogação em 1 e 3 anos, resultado e queda medianos,
   tempo sem capacidade, para o risco aberto alvo e para cada degrau da escada (F2-10);
5. a faixa de resultado esperado por idade (para o relatório mensal);
6. a comparação com o CDI, nas contas parada e histórica (F2-6);
7. posições admitidas (abertas) e sinais recusados, por motivo, por série;
8. a tabela de todas as regras da gramática por ativo — resultado, operações e peso final no Garimpo —, só
   como informação;
9. a mesma série sem a restrição de lote mínimo, como diagnóstico do custo do capital pequeno;
10. o diagnóstico de deriva (A10);
11. a listagem do histórico que o terminal MT5 guardou por conta própria (F3-2).

**D5. Determinismo.** O mesmo executor, no ambiente congelado no manifesto (versões de Python, NumPy e tzdata;
1 thread de BLAS; sem paralelismo nas reduções), dá a mesma saída byte a byte: JSON canônico, chaves ordenadas,
floats em repr de ida e volta, não finitos como texto declarado. A outra IA refaz do zero com o mesmo manifesto e
confere o hash da saída (F3-2); uma reimplementação independente compara campo a campo com tolerância 1e-12
(como o teste de F2-7). Rodar de novo sem mudar nada não
gasta fatia; repetir mais tarde, com mais dado, é ato novo e gasta a fatia seguinte.

**D6. Falha técnica.** A gravação é transacional: tudo vai para um pacote temporário, com fsync, e entra por
renomeação atômica junto com o contador de fatias; não existe arquivo parcial. Se o executor quebrar antes disso, nenhum resultado foi revelado:
corrige-se, a outra IA revisa a diferença, o hash novo vai ao proprietário e o ato recomeça sem gastar fatia. Se
alguma métrica já existir, a fatia está gasta (F3-4).

**D7. Limite declarado (RG-2).** A calibração (F2-9) certifica a taxa de falso positivo de uma simulação, desde que
valham as hipóteses dos mundos sintéticos; não é garantia sobre o mercado.
