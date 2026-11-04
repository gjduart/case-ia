"""Previsão de precisão e cobertura (previsao.txt) por intervalo de Jeffreys sobre o OOF da entrega.

Parte do pipeline de ./replicar.sh (importado por gerar_publicacao.py). Não lê nem grava
arquivos: devolve o texto de previsao.txt.

Cabeçalho (4 linhas lidas pela avaliação):
  precisao_prevista / precisao_intervalo = Jeffreys (média e quantis 2,5% e 97,5% de
  Beta(k + 1/2, n - k + 1/2)) com k e n = acertos e publicadas médios por repetição do
  OOF 5x5 da própria entrega, contados por questão DISTINTA (cada repetição cobre as
  mesmas 100 questões);
  cobertura_prevista = fração realmente publicada nas 559 questões fora da folha;
  cobertura_intervalo = Jeffreys da cobertura OOF (publicadas médias de 100 questões).
"""
from __future__ import annotations

from typing import Tuple

from scipy.stats import beta

from validacao import N_REPS


def jeffreys(k: float, n: float, nivel: float = 0.95) -> Tuple[float, float, float]:
    """Estima proporção e intervalo de credibilidade com priori de Jeffreys.

    Args:
        k: acertos (pode ser fracionário, média por repetição).
        n: total de tentativas (pode ser fracionário).
        nivel: nível de credibilidade do intervalo (padrão 0.95).

    Returns:
        Tupla (média a posteriori, limite inferior, limite superior) de Beta(k+0.5, n-k+0.5).
    """
    a, b = k + 0.5, n - k + 0.5
    al = (1 - nivel) / 2
    return float((k + 0.5) / (n + 1)), float(beta.ppf(al, a, b)), float(beta.ppf(1 - al, a, b))


def cabecalho(res: dict, cc: dict) -> dict:
    """Calcula os quatro números do cabeçalho de previsao.txt.

    Args:
        res: resumo do OOF (validacao.resumo) com K, N e T.
        cc: cobertura nas 559 (validacao.cobertura_559).

    Returns:
        Dicionário com prec, prec_lo, prec_hi, cob (publicada nas 559), cob_lo e cob_hi.
    """
    prec, lo, hi = jeffreys(res["K"] / N_REPS, res["N"] / N_REPS)
    _, clo, chi = jeffreys(res["N"] / N_REPS, res["T"] / N_REPS)
    return dict(prec=prec, prec_lo=lo, prec_hi=hi, cob=cc["p559"] / cc["n559"], cob_lo=clo, cob_hi=chi)


def texto(ctx, av: dict, cc: dict, areas: dict, cal_txt: str, mostrar_erros: str) -> str:
    """Monta o texto completo de previsao.txt.

    Args:
        ctx: contexto de dados (títulos das áreas).
        av: resultado de validacao.avaliar.
        cc: cobertura nas 559 (validacao.cobertura_559).
        areas: desempenho por área (validacao.por_area).
        cal_txt: relatório de calibração (calibracao.texto_relatorio).
        mostrar_erros: texto com as questões distintas erradas no OOF.

    Returns:
        Conteúdo de previsao.txt (cabeçalho de 4 linhas, linha em branco e explicação).
    """
    H, c = av["res"], cabecalho(av["res"], cc)
    reps = ", ".join(f"{a}/{b}" for a, b, _ in av["por_rep"])
    k_s, n_s = H["K"] / N_REPS, H["N"] / N_REPS
    por_area = "; ".join(f"{ctx.tax.titulo[r]}: OOF {v['oof_h']}/{v['oof_p']} publicadas de {v['oof_t']}; producao {v['prod_p']}/{v['prod_t']}" for r, v in areas.items())
    return f"""precisao_prevista: {c['prec']:.2f}
precisao_intervalo: {c['prec_lo']:.2f}-{c['prec_hi']:.2f}
cobertura_prevista: {c['cob']:.2f}
cobertura_intervalo: {c['cob_lo']:.2f}-{c['cob_hi']:.2f}

Previsao calculada (Jeffreys sobre o OOF da propria entrega), reproduzivel em ./replicar.sh (scripts/previsao.py)
-----------------------------------------------------------------------------------------------------
Entrega: A1 = roteador por area (qwen-LR de raiz, so texto) + S4 em Penal/Processo + consenso BM25 = qwen-LR em Legislacao Especial/Criminologia.
OOF 5x5 id-disjunto: {H['K']}/{H['N']} = {H['prec']:.3f}; por repeticao {reps}; k = {k_s:.1f} acertos e n = {n_s:.1f} publicadas por repeticao, contadas por questao DISTINTA (media das 5 repeticoes, pois cada repeticao cobre as mesmas 100 questoes).
precisao_prevista = Jeffreys (k+1/2)/(n+1) = ({k_s:.1f}+0,5)/({n_s:.1f}+1) = {c['prec']:.3f}; precisao_intervalo = quantis 2,5% e 97,5% de Beta(k+1/2, n-k+1/2) = {c['prec_lo']:.2f}-{c['prec_hi']:.2f} (Brown, Cai & DasGupta 2001, Statistical Science). Nada foi forcado a 0,95: e o que o calculo deu.
cobertura_prevista = fracao realmente publicada nas 559 = {cc['p559']}/{cc['n559']} = {c['cob']:.3f} (banco Q0001-Q0359: {cc['pb']}/{cc['nb']}; novas Q0360-Q0659: {cc['pn']}/{cc['nn']}); cobertura_intervalo = Jeffreys sobre a cobertura OOF da propria entrega ({n_s:.1f} de {H['T'] / N_REPS:.0f} questoes distintas) = {c['cob_lo']:.2f}-{c['cob_hi']:.2f}.
Questoes distintas erradas no OOF da entrega: {mostrar_erros}.

Vies de selecao (resultado historico do desenvolvimento, NAO recalculado aqui). A configuracao entregue foi escolhida entre varias olhando as mesmas 100 questoes, entao o OOF dela e otimista (Varma & Simon 2006, BMC Bioinformatics; Cawley & Talbot 2010, JMLR). Para descontar a escolha, durante o desenvolvimento avaliou-se o PROCEDIMENTO de escolha em 25 folds externos (as mesmas particoes 5x5) entre 35 candidatas: a A1 foi a escolhida em 25/25 folds e o limite conservador (Jeffreys) foi precisao 0,94 (0,82-1,00) e cobertura OOF 0,24 (0,16-0,33). O codigo dessa selecao aninhada e das demais rodadas de experimento foi removido do repositorio por simplicidade; os resultados estao em docs/ANEXO_EXPERIMENTOS.md. A coluna `confianca` herda o mesmo otimismo de selecao; a melhor estimativa agregada de precisao e a do cabecalho.

Como cheguei nos numeros
========================
Regra de publicacao (A1, sem LLM e sem rede em tempo de decisao; embeddings Qwen3 e sinteticos filtrados lidos do cache). Um roteador de raiz so de texto (LR sobre o embedding Qwen3, sem disciplina/assunto; treinado em sinteticos filtrados + descricoes + linhas da folha do treino) decide a area. Penal e Processo Penal: consenso NB hierarquico = BM25 = LSA no mesmo tema, estavel em >= 80% de 25 subamostras de 80% do treino (bagging do NB) e com score acima do piso do fold (quartil inferior do score do consenso, por CV interna so no treino). Legislacao Especial: BM25 (docs de tema enriquecidos com leis) = qwen-LR no tema da raiz roteada, norma citada concordante ou ausente, estavel em >= 80% de 25 subamostras do qwen-LR e score acima do piso da area. Criminologia: BM25 = qwen-LR + bagging + piso. A norma unica citada forca o tema (+2,8 no logit do score). Tudo e aprendido so no treino do fold externo.
Motivo da area: o estagio 1 do NB usa a disciplina_origem, que nas questoes novas (Q0360+) nunca e a de Legislacao Especial; sem o roteador, Legislacao Especial e Criminologia ficavam 100% em revisar (configuracao S4, anterior).

Precisao OOF bruta (otimista) = {H['K']}/{H['N']} = {H['prec']:.3f} (acertos/publicadas somados nas 5 repeticoes; pior repeticao {H['pior']:.3f}). Cobertura OOF {H['cob']:.3f}; cobertura nas 559 = {c['cob']:.3f} (a producao treina com as 92 nao-fora em vez de ~74 e inclui 300 questoes novas, fora da distribuicao da folha). A precisao nas novas nao e observavel.
Por area (raiz): {por_area}.

{cal_txt}

Limitacoes. (1) A folha so tem questoes Q0002-Q0356; a precisao nas novas (Q0360+) nao e observavel. (2) n pequeno de questoes distintas: com {n_s:.0f} publicadas distintas o intervalo e muito largo; a unica questao `fora` publicada (Q0043, fixacao da pena) tem rotulo `fora` na folha, mas o texto trata de dosimetria; o numero oficial segue o rotulo da equipe. (3) Selecao entre varias configuracoes nas mesmas 100 questoes (vies de selecao acima). (4) Sinteticos filtrados automaticamente, sem revisao humana. (5) Legislacao Especial (12 rotulos) e Criminologia (10) sao as areas mais fracas em dados: o OOF delas vem de poucas questoes distintas. (6) As linhas OOF usadas para calibrar a confianca vieram do protocolo original, cujos modelos podem ter treinado com ids do fold de teste externo (aproximacao do laco duplo); o vies residual favorece o otimismo.
"""
