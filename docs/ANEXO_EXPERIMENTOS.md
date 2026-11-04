# Anexo: detalhe dos experimentos (rodadas 1 a 4e, normalização)

Complementa o [relatório técnico](../relatorio/relatorio_tecnico.pdf) (fonte: `relatorio/relatorio_tecnico.tex`). Os números vêm de execuções de desenvolvimento (rodadas 1 a 4e) e de `previsao.txt`.

> **Nota (enxugamento do repositório, 2026-11-04).** O código dos experimentos (rodadas 2, 3, 4, 4b, 4c, 4d/S10, 4e/A4, seleção aninhada sobre ~35 candidatas, corpus OAB, embeddings e5 de questões e do BERT jurídico, configurações nunca entregues) foi **removido do repositório por simplicidade**: o repositório guarda só o que a entrega A1 precisa. Os números abaixo são os das execuções de desenvolvimento; o que continua reproduzível por `./replicar.sh` é a A1 (113/118 = 0,958; 103/559 publicadas; cabeçalho de `previsao.txt`). As menções a `CONFIG=...`, `--forcar-config` e a scripts de experimento descrevem o que existia durante o desenvolvimento e não são mais comandos do repositório. O limite conservador 0,94 (0,82–1,00) da seleção aninhada é, portanto, um resultado histórico, não recalculado.

 Protocolo comum: 5 repetições × 5 partes de teste sem questão repetida entre treino e teste sobre as 100 questões rotuladas (92 não-`fora` no treino); tudo que é aprendido (limiares, pesos, cortes) usa só o treino daquela rodada. Intervalos: Wilson 95%, com n = média de publicadas por repetição (as mesmas questões se repetem nas repetições). As configurações de cada rodada foram fixadas antes de rodar; todas foram avaliadas nas mesmas 100 questões, então o otimismo de seleção não está quantificado.

## 1. Histórico de versões e vazamento

| Versão | Publicadas (559) | Precisão declarada | Precisão honesta |
|---|---|---|---|
| Primeira (`../publicacao.csv`) | 67 | 57/58 = 0,983 | cerca de 0,91 |
| `final_otimo` original | 171 | 135/141 = 0,957 | 61/69 = 0,884 |
| R0 (consenso) | 150 | n/d | 0,891 |
| S4 sem normalizar | 77 | 73/78 = 0,936 | 0,936 (0,71–0,99) |
| S4 normalizada (entrega anterior) | 78 | 74/79 = 0,937 | previsão 0,91 (0,74–0,99); limite conservador aninhado sem S10*: 0,86 (0,68–0,97) |
| **A1 roteador por área (entregue)** | **103** | 113/118 = 0,958 | previsão **0,94 (0,82–1,00)**; limite aninhado sem S10*: 0,94 (0,82–1,00), sem descontar que a ideia nasceu de ver a S4 (seção 5d) |
| S10 (não adotada) | 62 | n/d | 1,000 bruto, regra criada após ver a Q0043; prevista com a S10 entre as candidatas: 0,87 (0,66–0,99) (histórico) |

As duas primeiras escolhiam o limiar nas outras repetições (`tr = ps[ps.rep != rep]`, `../scripts/gerar_publicacao.py:414`), que contêm as mesmas 100 questões: o teste vazava para a calibração. Nenhuma das ~40 configurações das rodadas 1–4c tem precisão ≥ 0,95 com mais de 100 publicadas; acima de 100 o melhor é R0 (0,891).

## 2. Rodada 1 (calibração id-disjunta)

| Config | Precisão (Wilson) | Cobertura | Por repetição |
|---|---|---|---|
| A cal. 5 feats, alvo 0,96 (regra antiga) | 0,886 (0,65–0,97) | 15,8% | 11/13 15/17 12/13 15/17 17/19 |
| B alvo 0,98 | 0,885 (0,61–0,97) | 12,2% | 9/11 11/12 11/12 11/13 12/13 |
| C alvo 1,00 | 0,909 (0,62–0,98) | 11,0% | 8/9 9/10 11/12 10/11 12/13 |
| D 4 feats, alvo 0,98 | 0,894 (0,63–0,98) | 13,2% | 10/11 11/13 13/14 12/14 13/14 |
| E só norma citada | 0,727 (0,43–0,90) | 11,0% | 8/11 em todas |
| F consenso NB=BM25=LSA | 0,891 (0,72–0,96) | 25,8% | 22/25 23/26 23/25 23/26 24/27 |
| H consenso OU norma | 0,842 (0,69–0,93) | 36,8% | 30/36 31/37 31/36 31/37 32/38 |
| I consenso E metadado | 0,896 (0,58–0,98) | 9,6% | 7/8 9/10 9/10 9/10 9/10 |
| J calibrador (score, lei, NB, BM25, LSA) | 0,897 (0,64–0,98) | 13,6% | 10/11 13/15 13/14 12/14 13/14 |

Rodada 2 (`experimentos_v2.py`, removido; 13 configurações): empilhadores e meta-modelos entre 0,50 e 0,88 de precisão, cobertura de 0,4% a 24,6%; detector de `fora` com AUC 0,40–0,49.

## 3. Rodada 3 (sintéticos, embeddings e5, OAB)

Acurácia de ramo top-1 OOF (92 × 5 = 460): NB 0,598; NB+sint 0,641; e5-LR 0,591; e5-centroide 0,543; BM25 e LSA 0,467; LSA+OAB 0,402.

| Config | Precisão (Wilson) | Cob. OOF | Por repetição | Prod. 559 (banco / novas) |
|---|---|---|---|---|
| R0 consenso NB=BM25=LSA | 0,891 (0,72–0,96) | 25,8% | 22/25 23/26 23/25 23/26 24/27 | 60/259 / 90/300 |
| R1 e5-LR com limiar em-fold | 0,875 (0,59–0,97) | 11,2% | 11/14 9/10 11/12 9/10 9/10 | 45 / 59 |
| R2 consenso(e5-LR, NB+sint) | 0,823 (0,70–0,90) | 52,0% | 44/54 43/50 44/51 42/54 41/51 | 104 / 148 |
| R3 consenso(e5-LR, centroide, NB+sint) | 0,860 (0,71–0,94) | 34,2% | 29/34 30/34 30/33 30/37 28/33 | 67 / 101 |
| R4 R3 + BM25 + LSA | 0,896 (0,64–0,98) | 13,4% | 11/13 12/13 13/14 12/14 12/13 | 30 / 51 |
| R5 R4 com LSA + OAB | 0,807 (0,52–0,94) | 11,4% | 9/11 9/11 10/12 9/12 9/11 | 29 / 50 |
| R6 R4 + margem e5-LR | 0,867 (0,33–0,99) | 3,0% | 3/4 2/2 2/2 3/4 3/3 | 17 / 30 |

Os votantes novos concordam nos mesmos erros: a acurácia de ramo não vira precisão de publicação. O corpus OAB (225 questões penais sem rótulo, decontaminadas) piorou o LSA e não entra em S4.

## 4. Rodada 4 (voltar ao R0 e tentar chegar a 95%)

Análise de erros do R0 (129 publicações OOF, 14 erros): 5 questões distintas, Q0043 (`fora`, 5 repetições), Q0017 (`fora`, 3), Q0202 (gabarito Erro de Tipo, previsto Dignidade Sexual, 4), Q0018 (1) e Q0188 (1).

Camadas, todas aprendidas só no treino da dobra: (i) bagging, NB retreinado em 25 subamostras de 80%, publica se o consenso se repete em ≥ τ dos sacos; (ii) veto OOD, distância de Mahalanobis (covariância compartilhada, Ledoit-Wolf) do embedding e5 ao cluster do tema previsto, vetando acima do percentil p das questões sem rótulo; (iii) piso de score, descartando o quartil inferior do score do consenso (corte por CV interna).

| Config | Precisão (Wilson) | Cob. OOF | Por repetição | Prod. 559 | Erros distintos |
|---|---|---|---|---|---|
| S0 = R0 | 0,891 (0,72–0,96) | 25,8% | 22/25 23/26 23/25 23/26 24/27 | 150 | Q0017 Q0018 Q0043 Q0188 Q0202 |
| S1 bagging τ 0,8 | 0,905 (0,71–0,97) | 21,0% | 19/21 18/21 19/21 20/22 19/20 | 134 | Q0017 Q0043 Q0188 Q0202 |
| S2 veto p15 | 0,886 (0,70–0,96) | 22,8% | 19/21 20/23 21/23 20/23 21/24 | 132 | Q0017 Q0043 Q0188 Q0202 |
| S3 S1+S2 | 0,891 (0,68–0,97) | 18,4% | 17/19 15/18 17/19 17/19 16/17 | 119 | idem |
| S4 S1 + piso do quartil (entregue; números após a normalização: 74/79 = 0,937, 15/16 na 4ª repetição, cob. 15,8%, prod. 78) | 0,937 (0,71–0,99) | 15,8% | 14/15 14/15 14/15 15/16 17/18 | 78 | Q0043 |
| S5 S3+S4 | 0,924 (0,67–0,99) | 13,2% | 12/13 11/12 13/14 11/12 14/15 | 71 | Q0043 |
| S6 Mondrian por raiz | 0,921 (0,66–0,99) | 12,6% | 12/13 12/13 11/12 10/11 13/14 | 75 | Q0043 |
| G τ0,8 p15 | 0,924 (0,67–0,99) | 13,2% | 12/13 11/12 13/14 11/12 14/15 | 71 | Q0043 |
| G τ1,0 p50 | 0,878 (0,54–0,98) | 8,2% | 8/9 7/8 6/7 6/7 9/10 | 44 | Q0043 |
| E1 τ1,0 p50 + metade superior | 0,872 (0,52–0,98) | 7,8% | 7/8 7/8 6/7 6/7 8/9 | 40 | Q0043 |
| E2 τ1,0 p70 + metade superior | 0,722 (0,27–0,95) | 3,6% | 2/3 4/5 2/3 2/3 3/4 | 30 | Q0043 |
| E3 τ1,0 + (norma OU e5-LR) | 0,906 (0,61–0,98) | 10,6% | 10/11 8/9 9/10 10/11 11/12 | 78 | Q0043 |
| E4 E3 + E1 | 0,872 (0,52–0,98) | 7,8% | 7/8 7/8 6/7 6/7 8/9 | 35 | Q0043 |

Grade completa (τ ∈ {0,8; 0,9; 1,0} × p ∈ {15, 30, 50, 70}) e linhas por raiz estão em `previsao.txt`. Regra de seleção fixa: passa se precisão pooled ≥ 0,95 e cada repetição ≥ 0,92 (margem: ≥ 0,97 e ≥ 0,93); se nada passa, publica o ponto de menor cobertura, que é E2. S4 foi escolhida pelo autor (decisão manual).

Por que o OOD não ajuda: AUC dos 8 `fora` contra a pontuação, Mahalanobis 0,380 e kNN-cos 0,331 (< 0,5: as `fora` ficam mais perto dos sintéticos que as demais). Q0043 e Q0017 não são vetadas em nenhum p. Por raiz em S4: Penal 50/55 OOF e 52 na produção; Processo Penal 21/21 e 25; Criminologia 2/2 e 0; Legislação Especial 0 e 0. Gap produção − OOF = −1,8 p.p. (13,8% contra 15,6%); um LR sobre o texto separa novas de banco com AUC 0,83; 85% das novas têm `assunto_origem` inédito e 3% têm disciplina inédita.

Não implementados: SetFit e calibração conformal (Learn-then-Test), porque com 100 rótulos o limite superior de erro seria vacuamente largo. Cross-encoders ficaram de fora por não poderem ser pré-calculados para questões novas.

## 5. Rodada 4b (embeddings melhores e votos extras)

Execução de desenvolvimento (101 s; código removido), 7 configurações fixadas antes de rodar, fora da regra de seleção.

| Voter sozinho (acurácia de ramo top-1, 460) | Acertos | Acurácia |
|---|---|---|
| e5-LR (referência) | 272 | 0,591 |
| qwen-LR (Qwen3-Embedding-0.6B) | 334 | 0,726 |
| legal-LR (stjiris legal BERTimbau) | 295 | 0,641 |
| BM25-kNN (k = 10) | 273 | 0,593 |

| Config (S4 E voto extra) | Precisão | Cob. OOF | Prod. 559 | Erros distintos |
|---|---|---|---|---|
| S4 (referência) | 0,936 | 15,6% | 77 | Q0043 |
| qwen | 0,914 | 11,6% | 68 | Q0043 |
| legal | 0,909 | 11,0% | 66 | Q0043 |
| BM25-kNN | 0,918 | 12,2% | 67 | Q0043 |
| vistos (tema com rótulo no treino) | 0,936 | 15,6% | 77 | Q0043 |
| BM25-kNN + vistos | 0,918 | 12,2% | 67 | Q0043 |

Critério de adoção (fixado antes): só troca S4 quem tiver precisão ≥ 0,936 com cobertura maior, ou ≥ 0,95. Nenhuma tem. Os votos extras cortam cobertura e nunca removem a Q0043. S4 permanece.

## 5b. Rodada 4c (consensos com qwen-LR)

Execução de desenvolvimento (157 s, offline; código removido), 3 configurações fixadas antes de rodar, mesmo protocolo aninhado e mesmos sacos de bagging da S4.

| Config | Precisão (Wilson) | Cob. OOF | Pior rep. | Prod. 559 | Erros distintos |
|---|---|---|---|---|---|
| S4 (referência) | 0,936 (0,71-0,99) | 15,6% | 0,93 | 77 | Q0043 |
| S7 consenso(qwen-LR, BM25, LSA) + bagging + piso | 0,887 (0,68-0,97) | 19,4% | 0,85 | 110 | Q0043, Q0202, Q0291 |
| S8 maioria 2-de-3 (NB, qwen-LR, BM25) + bagging + piso | 0,902 (0,77-0,96) | 40,8% | 0,89 | 197 | Q0043, Q0091, Q0202, Q0338, Q0339 |
| S9 S7 E NB concorda (4 votos) | 0,862 (0,58-0,97) | 11,6% | 0,78 | 61 | Q0043, Q0202 |

Regra fixa (precisão pooled >= 0,95 e cada repetição >= 0,92): nenhuma passa. S4 permanece e a entrega não muda. O qwen-LR é mais acurado sozinho (0,726 de ramo), mas ao entrar no consenso publica mais e traz erros novos além da Q0043; exigir o NB também (S9) corta cobertura sem remover a Q0202. A S8 tem a maior cobertura (197/559) e o maior limite inferior de Wilson, mas precisão 0,90. Os arquivos brutos (`cache/rodada4c_ex.pkl`, `cache/rodada4c_res.pkl`) foram removidos da entrega; `scripts/rodada4c.py` os regenera.

## 5c. Rodada 4d (S10: veto de "questão mista")

Execução de desenvolvimento (88 s, offline; código removido). Feature `r` (`questao_mista.py`, removido), sem rótulos e sem LLM: cada alternativa com mais de 3 palavras (após remover HTML) é classificada separadamente por Okapi BM25 sobre os documentos dos temas ativos de nível 1 e 2 (título + caminho + descrição sem "N questões no source"); r = nós top-1 distintos / alternativas; com menos de 2 alternativas textuais (Certo/Errado) r = NaN e a questão não é vetada. r é calculado uma vez para as 659 (NaN 165; r = 1 em 213; r ≥ 0,8 em 303). Mesmas partições e protocolo da S4. Configurações e regra de adoção fixadas antes de rodar: S10 = S4 E NÃO(r = 1,0); S10b = S4 E NÃO(r ≥ 0,8) só diagnóstico (não elegível); adotar S10 só se precisão pooled ≥ 0,95 e toda repetição ≥ 0,92, senão S4 e entrega idêntica.

| Config | Precisão (Wilson) | Cob. OOF | Por repetição | Distintas OOF | Prod. 559 (banco / novas) | Erros distintos |
|---|---|---|---|---|---|---|
| S4 (referência) | 0,936 (0,71–0,99) | 15,6% | 14/15 14/15 14/15 14/15 17/18 | 21 | 77 (35 / 42) | Q0043 |
| **S10 (r = 1 vetado, não adotada; valores pré-normalização)** | **1,000 (0,72–1,00)** | 9,8% | 9/9 9/9 10/10 9/9 12/12 | 15 | 62 (29 / 33) | nenhum |
| S10b (r ≥ 0,8, diagnóstico) | 1,000 (0,69–1,00) | 8,4% | 8/8 8/8 8/8 8/8 10/10 | 13 | 52 (23 / 29) | nenhum |

Custo do veto: remove 6 questões distintas publicadas no OOF (Q0043, Q0060, Q0221, Q0245, Q0246, Q0276), 5 delas corretas na S4 (24 das publicações OOF corretas); na produção, 25 das 106 publicações da S4 nas 659 (15 das 77 nas 559; 10 são da folha). Todas as vetadas têm r = 1. Em S10 a pior repetição é 1,00 e passa a regra; adotada. Nas 8 `fora` da folha, r = 1 em Q0002, Q0039, Q0043 e Q0303; Q0017 0,6; Q0087 0,8; Q0252 0,75; Q0291 NaN (4 de 7 com r definido; nas demais 30 de 70; AUC 0,607).

**Declaração de honestidade.** A regra foi concebida depois de inspecionar a Q0043, a única `fora` publicada (e único erro) da S4. Por isso o 1,000 do OOF não é evidência independente: é seleção nas mesmas 100 questões, com sinal fraco (AUC 0,607, só 7 `fora` com r definido). A S10 é mais conservadora (menos cobertura), mas a precisão nas novas segue não observável.

## 5d. Rodada 4e (roteador por área: A1, A2, A3)

`scripts/roteador_area.py` (offline, lê `cache/embeddings/qwen`; roda dentro do `replicar.sh`; a A1 é a única configuração que o repositório ainda executa). **Problema:** na S4, Criminologia (0/56 nas 559) e Legislação Especial (0/93) eram 100% `revisar`. Causa: as questões novas (Q0360+) nunca trazem a `disciplina_origem` "Direito Penal - Legislação Especial", então o estágio 1 do NB (classificador de raiz sobre o metadado de origem) as manda a Penal e o consenso NB = BM25 = LSA falha; Criminologia é teórica e tem só 10 rótulos (Legislação Especial, 12). Folha por raiz: Penal 40, Processo 30, Leg. Especial 12, Criminologia 10, `fora` 8.

Configurações e regra de adoção fixadas antes de rodar, mesmas partições 5×5 id-disjuntas, tudo aprendido só no treino da dobra externa:

- **Roteador** = LR (C = 10) só sobre o embedding Qwen3 (sem disciplina/assunto), alvo = raiz, treinada em sintéticos filtrados + descrições + linhas não-`fora` da folha do treino (peso 4).
- **Penal / Processo Penal:** exatamente a decisão da S4.
- **A1, Legislação Especial:** publica sse BM25 (documentos de tema enriquecidos com leis) top-1 = qwen-LR top-1 = tema (ambos restritos aos temas não duplicados da raiz roteada) E (`tema_lei_n1` unívoco da questão = tema OU nenhuma norma citada) E bagging (qwen-LR retreinado em 25 subamostras de 80% das linhas da folha do treino, sintéticos sempre incluídos; mesmos sacos da S4) τ = 0,8 E piso de score (quartil inferior do score, por CV interna e por área; cai para o quartil global com menos de 4 itens). Score = logit do blend 0,55 qwen-LR + 0,30 BM25 + 0,15 LSA no tema, mais 2,8 se a norma concorda.
- **A1, Criminologia:** publica sse BM25 top-1 = qwen-LR top-1 E bagging τ = 0,8 E piso por área.
- **A2** = A1 exigindo norma citada concordante em Legislação Especial. **A3** = A1 com τ = 1,0 nas duas áreas novas.
- **A4** (fixada depois da A1-A3, antes de ver seus resultados): roteador como na A1; em cada área, um LR um-contra-o-resto por tema (`OneVsRestClassifier(LogisticRegression(C=1, class_weight='balanced'))`) sobre os embeddings Qwen3, treino = sintéticos filtrados + descrições + linhas da folha do treino (as `fora` como negativas de todos os temas); limiar por tema escolhido em-fold por CV interna (menor limiar com precisão interna ≥ 0,95; temas com < 3 positivos usam o limiar global em-fold, mesma regra; sem limiar viável o tema nunca é aceito). Publica sse exatamente um tema da área passa o limiar E é o tema de consenso da área (S4 em Penal/Processo; BM25 = qwen-LR nas outras) E bagging τ = 0,8 (a condição vale em ≥ 20 de 25 subamostras, limiares fixos). Nenhum tema passa: revisar ("nenhum tema aceita"); dois ou mais: "conflito entre temas". Código: `python -B scripts/roteador_area.py --a4` (fora do `replicar.sh`).
- **Regra de adoção:** maior cobertura OOF entre as configurações com precisão pooled honesta ≥ 0,937 (a da S4), toda repetição ≥ 0,92 e estritamente mais publicações na produção que a S4 (78); senão a S4 fica e os arquivos de entrega ficam idênticos.

| Config | Por repetição | Pooled (Wilson) | Cob. OOF | Pior rep. | Prod. 559 (banco / novas) | Erros distintos |
|---|---|---|---|---|---|---|
| S4 | 14/15 14/15 14/15 15/16 17/18 | 74/79 = 0,937 (0,71–0,99) | 15,8% | 0,933 | 78 (35 / 43) | Q0043 |
| **A1** | 22/23 22/23 22/23 22/23 25/26 | 113/118 = 0,958 (0,79–0,99) | 23,6% | 0,957 | 103 (47 / 56) | Q0043 |
| A2 | idem | idem | idem | idem | 103 (47 / 56) | Q0043 |
| A3 | idem | idem | idem | idem | 103 (47 / 56) | Q0043 |
| A4 (não elegível) | 7/8 4/4 6/6 5/6 8/8 | 30/32 = 0,938 (0,55–0,995) | 6,4% | 0,833 | 55 (15 / 40) | Q0043, Q0202 |

Por raiz (OOF acertos/publicadas \| produção publicadas/total das 559 na raiz predita), S4: Penal 51/56 \| 52/230; Processo Penal 21/21 \| 26/180; Criminologia 2/2 \| 0/56; Legislação Especial 0/0 \| 0/93. A1 (= A2 = A3): Penal 46/51 \| 52/205; Processo Penal 21/21 \| 25/168; Legislação Especial 20/20 \| 22/122; Criminologia 26/26 \| 4/64. Roteador: acerta a raiz em 376/460 = 0,817 das linhas OOF não-`fora`. Erros distintos: só a Q0043 (publicada em Penal em 5 repetições), nas quatro configurações.

**A4:** por raiz OOF Penal 18/20, Processo 4/4, Legislação Especial 7/7, Criminologia 1/1; produção Penal 33, Processo 14, Legislação Especial 7, Criminologia 1. Rejeição de `fora` ("nenhum tema aceita"): 38 de 40 linhas `fora` (8 questões × 5 repetições) = 95,0%, mas a falsa rejeição nas não-`fora` é 391/460 = 85,0% e não houve nenhum conflito entre temas. Falha na regra (pior repetição 0,833 < 0,92; 55 < 78 publicações na produção): o limiar de precisão 0,95 por tema, estimado com 2 a 4 positivos por tema na CV interna, quase sempre é inviável e o detector rejeita quase tudo; entra Q0202 como novo erro. A rejeição de `fora` é útil, mas custa a cobertura.

**Decisão** (regra reaplicada a A1–A4 contra a S4; só A1–A3 são elegíveis): as três A passam a regra (0,958 ≥ 0,937, pior repetição 0,957 ≥ 0,92, 103 > 78) e decidem de forma idêntica em todas as 25 dobras e na produção (a norma e o τ = 1,0 não mudam nenhuma decisão: o gate BM25 = qwen-LR, o bagging e o piso já os absorvem); vale a A1, a mais simples (desempate declarado). Na produção a A1 perde 1 publicação da S4 nas 559 (roteada a outra área) e ganha 26 (Legislação Especial 22, Criminologia 4). **Honestidade:** a configuração foi concebida por causa da falha da S4 nessas duas áreas (não por erros), e foi avaliada nas mesmas 100 questões; com só 12 e 10 rótulos, o 20/20 e o 26/26 nas áreas novas vêm de poucas questões distintas e não provam precisão nas novas (a precisão nelas não é observável). **Correção de fundo:** rotular 30 a 50 questões por área e tirar o metadado de origem do estágio 1 do NB para as questões novas. Na API (`api/app.py`) o roteador e o qwen-LR exigem o Qwen3-Embedding-0.6B local (carregado na primeira chamada); a citação de norma é extraída do texto com as mesmas funções de `scripts/extrair_leis.py`.

## 6. Dados sintéticos e estrutura da árvore

- 1212 exemplos (1 descrição + 12 exemplos por tema-alvo, 101 temas) gerados por Claude Sonnet em 2026-11-03 só com a árvore e o texto da lei (`prompts/gerar_sinteticos.md`); nunca viram `questoes.jsonl` nem a folha. Filtro automático de ida e volta (e5-LR, tema no top-2; quase-duplicatas cos ≥ 0,98): 876/1212 = 72,3% mantidos (top-2 73,0%, top-1 52,2%, 9 quase-duplicatas). Revisei uma amostra aleatória de 30 (`cache/sinteticos_amostra_revisao.csv`): os 30 são coerentes com o tema.
- 12 nós duplicados sem filhos fundidos pela regra genérica de `scripts/estrutura.py` (Jaccard ≥ 0,6 mais o padrão "(Parte N)"): 6718, 6719 → 5087; 6720, 6721 → 5157; 6726, 6727 → 5199; 6732, 6733 → 5480; 6735 → 5543; 6736 → 5560; 6737, 6738 → 5572. Nenhum tem rótulo ouro.
- Citações de norma: `scripts/extrair_leis.py` reconstrói por regex e difere em 99 das 659 linhas do arquivo original (220 questões com norma contra 175).

## 7. Referências

Podolskiy et al., AAAI 2021 (Mahalanobis para OOD em transformers); Mahalanobis com covariância corrigida (arXiv 2609.00961); KNN-contrastivo para OOD, ACL 2022; Angelopoulos & Bates, Learn-then-Test (arXiv 2107.07511); Selective Conformal Risk Control (arXiv 2512.12844); SetFit.

## 8. Normalização dos dados (entrega final)

Auditoria dos dados do case: Q0099 traz HTML colado do MS Word (comentários condicionais `<!--[if ...]>...<![endif]-->` e `<style>`); Q0490 e Q0544 têm letras cirílicas/gregas no lugar de latinas (homóglifos); uma descrição em `temas.csv` contém palavra em árabe. `scripts/preprocessor.py` (`TextCleaner.clean`) remove os comentários condicionais, `<style>` e comentários HTML, mapeia os homóglifos para latino e remove trechos em escrita não latina; `scripts/taxonomy.py` (`build_pseudo_document`) remove a escrita árabe da `descricao`. A correção não depende do gabarito e é permitida pelo case (limpar e normalizar o texto). Efeito na S4: OOF de 73/78 para 74/79 (0,937), produção de 77 para 78 publicadas nas 559. Mudanças de decisão contra a versão anterior: Q0457 (revisar para publicar) e Q0231 (publicar para revisar, está na folha); Q0095, Q0182 e Q0285 só trocaram o tema sugerido em `revisar`. Os embeddings em cache não foram regenerados (a S4 não os usa). S10 após a normalização: 50/50 no OOF, 62/559 em produção.

## 9. S10 não adotada (e S4 substituída pela A1)

A S10 passa a regra de adoção fixada antes de rodar, mas a regra foi concebida depois de inspecionar a Q0043 (seção 5c). O autor optou por entregar a S4. Consequência para a previsão: o cabeçalho usa Jeffreys sobre o OOF da própria S4 e o limite conservador é a seleção aninhada **sem** as S10* (candidatas S10 e S10b excluídas por `selecao_aninhada.executar` quando a entrega não é S10*). (`CONFIG=S10`, comando existente só no desenvolvimento) reproduzia a S10 com cabeçalho aninhado incluindo a S10 (números podem diferir ligeiramente do histórico, que é pré-normalização). A entrega padrão passou a ser a A1 (seção 5d); o cabeçalho e o limite aninhado (sem S10*) são calculados com a A1 e a S4 entre as candidatas.

## 10. Fluxo de validação e tabela completa

```mermaid
flowchart LR
  F[100 rótulos, 92 não-fora] --> CV[5x5 dobras id-disjuntas]
  CV --> TR[treino, limiares e calibração só no treino]
  TR --> TE[decisão aplicada à dobra de teste]
  TE --> OOF[OOF da A1: 113/118, Jeffreys]
  CV --> NI[seleção aninhada: 35 candidatas sem S10*, só ids fora do teste]
  NI --> LC[limite conservador 0,94 (0,82-1,00)]
```

As tabelas das seções 1 a 5d listam as cerca de 50 configurações testadas; os valores brutos estão em `previsao.txt`.

## Seleção aninhada e previsão de precisão (`scripts/previsao.py` para o cabeçalho; o código da seleção aninhada foi removido)

Motivo: a configuração entregue foi escolhida entre várias olhando as mesmas 100 questões; o OOF bruto (113/118) é otimista (Varma & Simon 2006, *BMC Bioinformatics*; Cawley & Talbot 2010, *JMLR*, "On over-fitting in model selection and subsequent selection bias in performance evaluation").

Método. Em cada uma das 25 dobras externas (mesmas partições 5×5), a regra de escolha fixa do projeto (`experimentos_v4.selecionar`: nível 2 = pooled ≥ 0,97 e cada repetição ≥ 0,93; nível 1 = ≥ 0,95 e ≥ 0,92; dentro do nível, maior cobertura; sem nenhuma, a mais precisa com cobertura ≥ 10%) escolhe entre 35 candidatas (S1–S6, grade τ × p, E1–E4, S7–S9, rodada 4b, A1–A3; **sem S10 e S10b**) usando só as linhas OOF dos ids fora da dobra de teste; a escolhida é aplicada à dobra de teste. Resultado: A1 escolhida em 25/25 dobras; por repetição 22/23, 22/23, 22/23, 22/23, 25/26; k = 22,6 e n = 23,6 por questão distinta (28 distintas na união); erro distinto: Q0043 (×5). Jeffreys (Brown, Cai & DasGupta 2001, *Statistical Science*): 0,94 (0,82–1,00); cobertura OOF aninhada 0,24 (0,16–0,33). **Este é o limite conservador** (coincide com o cabeçalho porque a A1 foi escolhida em todas as dobras), não o cabeçalho; ele desconta a escolha entre as candidatas, **não** desconta que a A1 foi desenhada depois de ver a falha da S4 em Legislação Especial e Criminologia. Para comparação, com a S4 como entrega o mesmo procedimento (sem A1–A3) dava 0,86 (0,68–0,97), S4 escolhida em 20/25 dobras.

Cabeçalho do `previsao.txt` (A1): Jeffreys sobre o OOF da própria A1, k = 113/5 = 22,6 e n = 118/5 = 23,6 questões distintas por repetição: precisão (k+½)/(n+1) = 0,94, intervalo de Beta(k+½, n−k+½) = 0,82–1,00; cobertura prevista = 103/559 = 0,18, intervalo = Jeffreys da cobertura OOF (23,6 de 100) = 0,16–0,33. Histórico: a S4 declarava 0,91 (0,74–0,99) e a versão que entregava a S10, 0,87 (0,66–0,99). Nada foi forçado a 0,95.

Limitação (aproximação). As linhas OOF dos ids de treino de cada dobra vêm do protocolo original, cujos modelos podem ter treinado com os ids da dobra de teste externa; a versão estrita (pipeline inteiro num laço duplo) não cabe no tempo do `replicar.sh`. O viés residual favorece o otimismo.

Modelos de linguagem e a regra do case (seção 4). Claude Sonnet escreveu sintéticos e descrições só a partir da árvore e da lei (nunca viu as questões); o Qwen3-Embedding-0.6B apenas converte texto em vetor, em preparação (`scripts/embeddings.py`, cache em `cache/embeddings/qwen/`); a qwen-LR da A1 treina com sintéticos filtrados, descrições e folha. A decisão é de regressões logísticas próprias e regras de consenso; `replicar.sh` roda offline sem carregar modelo de linguagem. Na API, o embedding é a etapa `POST /ingerir`; `POST /classificar` usa só o vetor guardado (garantido em `tests/test_api_ingestao.py`). Sem rede neural: a S4 (78 publicadas, 0,937 na validação) existiu durante o desenvolvimento como `CONFIG=S4` e não é mais executável no repositório.


## Arquiteturas das primeiras versões (GCN, BiGRU, TextCNN, SAT/z3, árvores, self-training, RL)

Antes da versão final testei 26 modelos de famílias diferentes. Os scripts dessa fase ficaram fora do repositório (histórico local). A métrica é o acerto do ramo (não a precisão das publicadas):
- NB hierárquico: 0,467 (rep 0) / 0,543 (5×5);
- self-training: até 0,467;
- SAT neural/regras (z3): 0,34–0,38;
- GCN e ExtraTrees: 0,304;
- TextCNN: 0,294;
- e5 congelado: 0,20–0,28;
- BiGRU: 0,185;
- MLP: 0,033.

Bandit/RL, mega-ensemble e pseudo-rótulos do Senado foram avaliados com a validação que tinha vazamento; os números da época não são comparáveis. Nenhuma superou o NB condicionado pela hierarquia com 92 rótulos.
