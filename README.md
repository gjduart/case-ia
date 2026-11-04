# Classificação de questões jurídicas com publicação seletiva

Minha solução para o case **"Classificação de tópicos"** (processo seletivo, área de IA, método VDE).

**Relatório (4 páginas, com o memo para o professor):** [`relatorio/relatorio_tecnico.pdf`](relatorio/relatorio_tecnico.pdf)
**Experimentos em detalhe:** [`docs/ANEXO_EXPERIMENTOS.md`](docs/ANEXO_EXPERIMENTOS.md)

---

## 1. Do que se trata

Um sistema que lê cada questão de concurso (Penal, Processo Penal, Legislação Penal Especial e Criminologia), sugere o **tema** dela na árvore de temas da plataforma e decide:

- **`publicar`**: a questão vai direto para o aluno, no tema sugerido;
- **`revisar`**: a questão vai para a fila do professor, com o tema sugerido e o motivo.

O sistema só publica sozinho quando tem segurança. Na dúvida, manda para o professor.

## 2. O problema que ele resolve

Parte das questões da plataforma foi classificada automaticamente e de forma errada. Isso criou **"lixeiras"**: temas genéricos que acumularam centenas de questões sem relação com eles. Quando a classificação está errada, o aluno estuda a matéria errada.

**A régua do case:** das questões publicadas, **pelo menos 95% precisam estar certas**. Respeitada essa condição, quanto mais publicar, melhor; o resto vai para revisão. Na hora de classificar, **nenhum modelo de linguagem pode ser chamado**.

## 3. Resultado

| | Valor |
|---|---|
| Questões avaliadas (fora da folha-verdade) | 559 |
| **Publicadas** | **103 (18%)**: Penal 52, Processo Penal 25, Legislação Especial 22, Criminologia 4 |
| **Precisão na validação** (100 questões rotuladas, sem vazamento) | **113/118 = 0,958**; único erro: a Q0043 |
| **Precisão prevista** (`previsao.txt`) | **0,94 (intervalo 0,82–1,00)** |
| Cobertura prevista | 0,18 (intervalo 0,16–0,33) |
| Meta de 95% nas questões novas | **não demonstrada**: não há rótulos para medir as 300 questões de concursos recentes |

A única questão errada na validação, a Q0043, foi marcada `fora` pela equipe, mas trata de dosimetria da pena, tema que existe na árvore. O relatório e o memo pedem ao professor que confira esse rótulo.

## 4. Como reproduzir

Os três caminhos geram o mesmo `publicacao.csv`, com md5 `0d9329da890c4816850747cdf3855f99`, e o mesmo cabeçalho de `previsao.txt` (`0.94`, `0.82-1.00`, `0.18`, `0.16-0.33`). Nenhum deles usa internet nem modelo de linguagem.

**a) Um comando** (Python 3.14; cerca de 3,5 min em CPU):
```bash
pip install -r requirements.txt
./replicar.sh
```

**b) Docker** (ambiente pronto, rede desligada; cerca de 4 min; testado com Docker 29.9 e Compose 5.7):
```bash
export DOCKER_UID=$(id -u) DOCKER_GID=$(id -g)   # arquivos gerados ficam com o seu usuário
docker compose run --rm replicar
```

**c) Notebook, passo a passo:** [`replicacao_final_otimo.ipynb`](replicacao_final_otimo.ipynb) **executa de verdade**, célula a célula, tudo o que dá para recomputar offline: normalização do texto, árvore e fusão de duplicados, extração de leis (comparada ao cache), o filtro dos sintéticos (`filtrar_sinteticos.py` em pasta temporária, comparado byte a byte ao cache), os três leitores, o roteador, o OOF 5x5 completo, a calibração, o Jeffreys e, no fim, o `main()` de `gerar_publicacao.py` inteiro, do zero, com conferência de md5 (OK ou DIFERENTE; só copia para a pasta se for OK). Não há atalhos nem substituição de funções. Fora da execução, por escolha ou por natureza: a geração dos sintéticos (LLM, dado versionado), os vetores Qwen3/e5 (preparação em `scripts/embeddings.py`, lidos do cache) e a consulta ao Senado (flag `RODAR_REDE = False`, precisa de internet). Tempo medido: cerca de 6 min em 12 CPUs (`jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 replicacao_final_otimo.ipynb`). Requer `pip install -r requirements-dev.txt`.

## 5. Como funciona

```text
questão → limpeza do texto → roteador por área ─┬─ Penal / Processo Penal:   NB = BM25 = LSA
                                                └─ Leg. Especial / Criminologia: BM25 = regressão logística sobre vetores
        → a resposta se mantém em 25 retreinos? → acima do piso de confiança? → publicar | revisar
        → calibração: a confiança vira "chance de o tema estar certo"
```

1. **Limpeza do texto:** remove HTML do Word colado e troca letras cirílicas que se parecem com latinas.
2. **Árvore de temas:** funde 12 temas duplicados (por exemplo, "Teoria da Pena (Parte 1)") e prevê só o nível 1 (ver seção 6).
3. **Roteador por área:** uma regressão logística escolhe a área pelo texto, sem usar o metadado de origem, que nas questões novas nunca indica Legislação Especial.
4. **Leitores independentes:**
   - em Penal e Processo Penal: Naive Bayes hierárquico, busca BM25 e análise semântica latente (LSA);
   - nas áreas novas: BM25 e uma regressão logística sobre os vetores de texto.
5. **Decisão:** publica só quando os leitores concordam, a resposta se mantém ao retreinar o modelo 25 vezes com subconjuntos dos rótulos, e o escore passa de um piso.
6. **Calibração:** a `confianca` passa a significar a chance de o tema estar certo.
7. **Previsão:** a precisão prevista usa a correção de Jeffreys para amostra pequena.

Validação: as 100 questões rotuladas são divididas em 5 partes; treino com 4 e testo na quinta, até cada parte ser teste uma vez, e repito 5 vezes embaralhando. Cada questão é testada por um modelo que não a viu.

## 6. Por que só o nível 1

**O que o case pede:** "o tema central" de cada questão, um `tema_id` de `temas.csv`; qualquer nível vale, e **não há exigência de chegar ao nível 2**.

A régua conta como certo publicar no tema da equipe ou num nó do mesmo ramo (pai ou filho dele), e conta como **erro** publicar num subtema irmão. Por isso publico no nível 1: nas 39 questões da folha com gabarito em subtema, o tema-pai conta como acerto, e evito o erro entre irmãos.

O custo é que quem estuda direto pelo subtema não encontra a questão. Com mais tempo, eu desceria ao subtema só quando a lei e o artigo citados apontam um subtema único.

## 7. Onde entra (e onde não entra) modelo de linguagem

A seção 4 do case permite usar modelo de linguagem **antes** da decisão, documentado e versionado (gerar embeddings, enriquecer temas, propor exemplos revisados), e proíbe que ele escolha, vote ou valide o tema.

1. **Claude Sonnet** gerou exemplos e descrições de cada tema a partir **só da árvore e do texto da lei**, sem ver as questões. Esse material está guardado como dado (`cache/sinteticos/`, prompt em `prompts/gerar_sinteticos.md`). Passou por um filtro automático (876 de 1.212 mantidos) e revisei uma amostra de 30, todos coerentes com o tema. Serve só como dado de treino da regressão logística.
2. **Qwen3-Embedding-0.6B** só transforma texto em vetor de números, numa etapa de preparação. Os vetores ficam em `cache/embeddings/qwen/`. Ele não gera texto e não vê os temas.
3. **A decisão é das minhas regressões logísticas e regras de consenso.** O `replicar.sh` não carrega nenhum modelo de linguagem.

## 8. Estrutura da pasta

```text
replicar.sh                          comando único: gera publicacao.csv e previsao.txt
publicacao.csv, previsao.txt         entregáveis do case
questoes.jsonl, temas.csv, folha_verdade.csv   anexos do case (entradas)
relatorio/relatorio_tecnico.{pdf,tex} relatório técnico (seções a–e + memo), template SBC
docs/ANEXO_EXPERIMENTOS.md           tudo o que testei, com números
replicacao_final_otimo.ipynb         notebook de replicação passo a passo
scripts/
  gerar_publicacao.py                orquestra a entrega
  dados.py                           carrega e prepara dados, árvore, leitores e vetores
  roteador_area.py                   roteador por área e regra das áreas novas
  decisao.py                         regra de Penal e Processo Penal (consenso + estabilidade + piso)
  validacao.py                       validação cruzada 5×5
  calibracao.py, previsao.py         confiança calibrada e previsão de Jeffreys
  models.py, bm25.py, taxonomy.py, preprocessor.py, estrutura.py   leitores, árvore, limpeza, fusão de duplicados
  extrair_leis.py                    leis citadas em cada questão (roda no replicar.sh)
  atualizar_normas.py, embeddings.py, filtrar_sinteticos.py        preparação (ver seção 9)
prompts/                             prompt versionado e temas usados na geração dos exemplos
cache/                               artefatos versionados (ver seção 9)
api/, tests/, Dockerfile, docker-compose.yml   extras (ver seção 11)
requirements.txt                     dependências do replicar.sh (versões fixadas)
requirements-dev.txt, requirements-api.txt     notebook/testes e API
```

O código tem docstrings no padrão Google, em português.

## 9. Artefatos em `cache/` e como regenerá-los

Os artefatos já estão versionados; o `replicar.sh` só os lê. Rode a preparação apenas se quiser reconstruí-los.

| Artefato | O que é | Como foi gerado |
|---|---|---|
| `normas_senado.json` | metadados das leis citadas | `scripts/atualizar_normas.py` (usa a API de Dados Abertos do Senado) |
| `lei_tema.csv` | lei → tema | montado automaticamente a partir do Senado e dos títulos dos temas |
| `leis_por_questao.csv` | leis citadas em cada questão | `scripts/extrair_leis.py` (refeito a cada `replicar.sh`) |
| `sinteticos/` | 1.212 exemplos e descrições por tema | Claude Sonnet, prompt em `prompts/gerar_sinteticos.md` |
| `sinteticos_filtrados.jsonl` | 876 exemplos que passaram no filtro automático | `scripts/filtrar_sinteticos.py` |
| `sinteticos_amostra_revisao.csv` | amostra de 30 revisada por mim | `scripts/filtrar_sinteticos.py` (nunca sobrescreve a revisão) |
| `embeddings/qwen/` | vetores Qwen3-Embedding usados pela entrega | `scripts/embeddings.py` |
| `embeddings/e5/` | vetores usados só pelo filtro dos exemplos | `scripts/embeddings.py --modelo e5` |
| `modelo_producao.pkl` | modelo ajustado, lido pela API | `replicar.sh`; fora do git (cerca de 430 MB) |

```bash
python -B scripts/atualizar_normas.py          # usa internet
python -B scripts/embeddings.py                # baixa o Qwen3-Embedding na primeira vez
python -B scripts/embeddings.py --modelo e5
python -B scripts/filtrar_sinteticos.py
```

## 10. Decisões tomadas à mão e limitações

**Decisões:**
1. **Roteador por área (entrega atual).** Sem ele, Criminologia e Legislação Especial ficavam com zero publicações. Tive a ideia depois de ver esse problema, e essas áreas têm só 10 e 12 rótulos; a estimativa nelas é otimista.
2. **Veto de "questão mista" descartado.** Ele levava a validação a 100%, mas nasceu da análise da própria Q0043.
3. **Normalização dos dados** encontrados com defeito (Q0099, Q0490, Q0544 e uma palavra em árabe em `temas.csv`).
4. **Revisão dos sintéticos:** uma amostra de 30, não os 876 um a um.
5. **Q0043:** mantida com o rótulo `fora` da equipe no número oficial.
6. **Limiares fixados antes de rodar** e nunca ajustados depois de ver o resultado.

**Limitações:**
- A precisão nas 300 questões novas não pode ser medida, por falta de rótulos.
- Todas as configurações foram comparadas nas mesmas 100 questões; o relatório explica como esse otimismo foi descontado.
- O sistema não emite `fora`: com só 8 exemplos, nenhum detector teve sinal. Na dúvida, a questão vai para `revisar`.
- O código dos experimentos descartados foi removido para deixar o repositório simples; os resultados estão em `docs/ANEXO_EXPERIMENTOS.md`.

## 11. Extras (não fazem parte da entrega pedida)

**API de demonstração (FastAPI).** Separa a geração do vetor (`POST /ingerir`) da classificação (`POST /classificar`), que nunca chama modelo de linguagem. Precisa do `modelo_producao.pkl` (gerado pelo `replicar.sh`) e dos pesos do Qwen3-Embedding já baixados.
```bash
pip install -r requirements-api.txt
python -m uvicorn api.app:app --port 8000      # documentação em http://localhost:8000/docs
# ou: docker compose up api
```

**Testes:** `python -m pytest tests -q`.

## 12. Entregáveis do case

| Pedido no case | Onde está |
|---|---|
| `publicacao.csv` | raiz do repositório |
| `previsao.txt` | raiz do repositório |
| Código, com um comando que regenera a entrega sem modelo de linguagem | `replicar.sh` e `scripts/` |
| Relatório (até 4 páginas, seções a–e e memo) | `relatorio/relatorio_tecnico.pdf` |
