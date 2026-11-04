# Prompt versionado — geração de exemplos sintéticos e descrições de temas

**Base no case (PDF, seção 4):** um modelo de linguagem *pode*, antes da decisão e de forma documentada e versionada, "enriquecer a descrição dos temas, propor termos ou exemplos que você revisa e guarda como dado". Ele *não pode* escolher tema, votar ou validar as questões avaliadas.

**Gerador:** Claude Sonnet, rodando como subagente do Claude Code em 2026-11-03, em 4 lotes paralelos (A–D), definidos em `prompts/temas_alvo.jsonl`.

**Isolamento:** o gerador recebeu **só** a árvore de temas (`prompts/temas_alvo.jsonl`, derivado de `temas.csv`) e o texto oficial das leis (parse de normas.leg.br em `cache/lei_capitulos.jsonl`, arquivo que não é lido pela entrega e foi retirado do repositório no enxugamento de 2026-11-04). Ele **não teve acesso** a `questoes.jsonl`, `folha_verdade.csv`, `publicacao.csv` nem a nenhum arquivo derivado das questões.

**Revisão:** filtro automático de consistência de ida e volta (`scripts/filtrar_sinteticos.py`) e uma amostra aleatória para conferência humana em `cache/sinteticos_amostra_revisao.csv`.

---

## Instrução dada ao gerador (por tema)

Você é um elaborador de questões de concursos jurídicos brasileiros (bancas como CESPE/Cebraspe, FGV, FCC, Vunesp) de Direito Penal, Processo Penal, Legislação Penal Especial e Criminologia.

Para o **tema** informado (caminho na árvore, descrição, subtemas e temas irmãos):

1. **Descrição contrastiva** (registro `tipo="descricao"`):
   - `inclui`: 3 a 6 frases sobre o que pertence ao tema, cobrindo todos os subtemas;
   - `exclui`: o que **não** pertence ao tema e vai para um irmão específico (cite o irmão pelo nome);
   - `termos`: 15 a 30 termos/expressões técnicas típicos do tema (nomes de crimes, institutos, artigos e leis);
   - `artigos`: lista de dispositivos legais centrais (ex.: "CP art. 312", "CPP art. 155").
2. **12 questões sintéticas** (registros `tipo="exemplo"`) que um especialista classificaria **neste tema** (ou em um de seus subtemas), nunca num irmão:
   - distribua entre os subtemas (indique `subtema` quando houver);
   - varie o estilo: cerca de 5 casos concretos narrativos que **não citam** o nome do instituto nem o artigo; 3 de letra de lei; 2 de jurisprudência (STF/STJ, sem inventar número de julgado); 2 de assertiva Certo/Errado;
   - formato: `enunciado` (1 a 5 frases), `comando` (ex.: "Assinale a alternativa correta."), `alternativas` (objeto A–E, ou "Certo"/"Errado" para assertivas);
   - use a lei vigente; quando houver trecho de lei fornecido, baseie-se nele;
   - **não** gere questões ambíguas que caberiam igualmente num irmão: se o assunto fica na fronteira, escolha outro.

## Formato de saída
Um arquivo por tema, `cache/sinteticos/tema_<tema_id>.jsonl`, em UTF-8, com um JSON por linha:

```
{"tema_id": 5157, "tipo": "descricao", "inclui": "...", "exclui": "...", "termos": ["..."], "artigos": ["..."]}
{"tema_id": 5157, "tipo": "exemplo", "subtema": "Furto", "estilo": "caso_concreto", "enunciado": "...", "comando": "...", "alternativas": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}}
```
