#!/usr/bin/env python3
"""Regeneração de publicacao.csv e previsao.txt (sem LLM, sem rede): configuração A1.

Passo [3/4] de ./replicar.sh. Entradas: questoes.jsonl, temas.csv, folha_verdade.csv e
cache/ (embeddings Qwen3, sintéticos filtrados, normas). Saídas: publicacao.csv,
previsao.txt (caminhos em --saida e --previsao) e cache/modelo_producao.pkl (lido só
pela API).

Etapas: [1/4] carrega os dados; [2/4] roda o OOF 5x5 e a produção (roteador por área,
S4 em Penal/Processo, consenso BM25 = qwen-LR em Legislação Especial/Criminologia);
[3/4] calibra a coluna `confianca` (OOF aninhado id-disjunto; não muda as decisões);
[4/4] grava publicacao.csv, o modelo de produção e previsao.txt.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibracao as cb  # noqa: E402
import decisao  # noqa: E402
import previsao  # noqa: E402
import roteador_area as ra  # noqa: E402
import validacao as val  # noqa: E402
from dados import BASE_DIR, Contexto  # noqa: E402

NOME = "A1 roteador por area (qwen-LR de raiz; S4 em Penal/Processo; consenso BM25=qwen-LR em Leg.Esp./Crim.)"
NV = {"nb0": "NB", "bm": "BM25", "ls": "LSA"}


def motivo_revisar(ctx: Contexto, r, need: int) -> str:
    """Explica por que uma questão não foi publicada.

    Args:
        ctx: contexto de dados.
        r: linha (namedtuple) da produção com pub False.
        need: sacos de bagging necessários (20).

    Returns:
        Texto "revisar: ...; tema sugerido pelo ..." com todos os motivos aplicáveis.
    """
    mot = []
    if r.nova_area:
        if not r.gate_bm:
            mot.append("sem consenso BM25 = qwen-LR na area roteada (" + ctx.tax.titulo[int(r.rt)] + ")")
        else:
            if not r.lawok:
                mot.append("a norma citada aponta outro tema")
            if r.cnt < need:
                mot.append(f"instavel sob bagging ({r.cnt}/{decisao.B_BAGS} sacos < {need})")
            if r.score < r.qg:
                mot.append("score do consenso abaixo do piso da area (quartil inferior)")
        return "revisar: " + "; ".join(mot) + "; tema sugerido pelo roteador por area"
    falt = [NV[v] + " diverge" for v in ("nb0", "bm", "ls") if not getattr(r, "ag_" + v)]
    if falt:
        return "revisar: sem consenso NB=BM25=LSA (" + "; ".join(falt) + "); tema sugerido pelo ensemble"
    if r.cnt < need:
        mot.append(f"instavel sob bagging ({r.cnt}/{decisao.B_BAGS} sacos < {need})")
    if r.score < r.qg:
        mot.append("score do consenso abaixo do piso (quartil inferior)")
    return "revisar: " + "; ".join(mot) + "; tema sugerido pelo ensemble"


def salvar_modelo_producao(ctx: Contexto, prod: pd.DataFrame, mod_area: dict, cal: dict, need: int, caminho: Path) -> None:
    """Ajusta e grava o pipeline de produção em toda a folha (lido por api/app.py).

    NB cheio e 25 sacos (mesmas sementes da linha "prod" do OOF), piso da S4, calibrador
    e os modelos do roteador/áreas novas.

    Args:
        ctx: contexto de dados.
        prod: DataFrame da produção (para o piso da S4).
        mod_area: modelos de roteador_area.decidir_fold(guardar=True).
        cal: calibrador de confiança.
        need: sacos de bagging necessários.
        caminho: arquivo de saída (pickle).
    """
    clf = ctx.clf
    lab = ctx.sup[ctx.sup.y != -1]
    rng = np.random.RandomState(decisao.SEED_BAG + 999)
    bags = []
    for _ in range(decisao.B_BAGS):
        idx = rng.choice(len(lab), int(round(decisao.FRAC_BAG * len(lab))), replace=False)
        bags.append(clf.fit_hierarchy(lab.iloc[idx]))
    st = dict(nome=NOME, cal=cal, B=decisao.B_BAGS, need=need, full=clf.fit_hierarchy(lab), bags=bags, qg=float(prod.qg_s4.iloc[0]),
              rota=dict(mod=mod_area, need=need, B=decisao.B_BAGS, areas_s4=list(ra.AREAS_S4), areas_novas=list(ra.AREAS_NOVAS), law_ok_leg="ou_sem_norma"))
    caminho.write_bytes(pickle.dumps(st))


def main():
    """Executa o pipeline completo e grava publicacao.csv, previsao.txt e o modelo de produção.

    Opções: --saida (publicacao.csv), --previsao (previsao.txt), --modelo (pickle da API).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(BASE_DIR / "publicacao.csv"))
    ap.add_argument("--previsao", default=str(BASE_DIR / "previsao.txt"))
    ap.add_argument("--modelo", default=str(BASE_DIR / "cache" / "modelo_producao.pkl"))
    args = ap.parse_args()

    print("[1/4] Carregando dados (texto normalizado, taxonomia, BM25/LSA, embeddings Qwen3)...")
    ctx = Contexto("qwen")
    print(f"      nos duplicados mesclados (scripts/estrutura.py): {len(ctx.dup)}; sinteticos mantidos pelo filtro: {int(ctx.keep.sum())}/{len(ctx.keep)}")

    print("[2/4] OOF 5x5 + producao (roteador por area; S4 em Penal/Processo; consenso BM25=qwen-LR nas areas novas; offline, ~5 min em CPU)...")
    ex, mod_area = val.executar(ctx)
    av = val.avaliar(ex)
    cc = val.cobertura_559(ctx, av["prod"])
    H = av["res"]
    areas = val.por_area(ctx, av)
    need = decisao.sacos_necessarios()
    print(f"      OOF: {H['K']}/{H['N']} = {H['prec']:.3f} (pior repeticao {H['pior']:.3f}); cobertura OOF {H['cob']:.3f}; publicadas nas 559: {cc['p559']}/{cc['n559']}")

    print("[3/4] Calibrando a confianca (OOF aninhado id-disjunto; decisoes nao mudam)...")
    oof_cal, cal = cb.oof_aninhado(ex, val.publicar, val.N_REPS, val.N_FOLDS)
    cal_txt = cb.texto_relatorio(cb.relatorio(oof_cal))
    prod = av["prod"].copy()
    prod["cal"] = cb.aplicar(cal, cb.feats_df(prod))

    print("[4/4] Gravando publicacao.csv, modelo de producao e previsao.txt...")
    linhas = [dict(id=r.id, decisao="publicar" if r.pub else "revisar", tema_central_id=int(r.pred), extras_ids="",
                   confianca=round(float(np.clip(r.cal, 0, 1)), 4), motivo="" if r.pub else motivo_revisar(ctx, r, need))
              for r in prod.itertuples()]
    pub_df = pd.DataFrame(linhas)
    pub_df.to_csv(args.saida, index=False, encoding="utf-8")
    salvar_modelo_producao(ctx, prod, mod_area, cal, need, Path(args.modelo))
    erros = ", ".join(f"{q} (x{c})" for q, c in sorted(av["oof"][av["oof"].pub & (av["oof"].ramo != 1)].groupby("id").size().to_dict().items())) or "nenhuma"
    Path(args.previsao).write_text(previsao.texto(ctx, av, cc, areas, cal_txt, erros), encoding="utf-8")
    print(f"Arquivos: {args.saida} ({len(pub_df)} linhas), {args.previsao}")


if __name__ == "__main__":
    main()
