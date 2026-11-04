#!/usr/bin/env python3
r"""Etapa de PREPARAÇÃO (não roda no replicar.sh): gera os embeddings em cache/embeddings/<modelo>/.

Embeddings são pré-processamento (só texto para vetor; modelo local, sem LLM generativo).
Dois modelos:
  qwen  = Qwen/Qwen3-Embedding-0.6B (vetores LIDOS pela entrega): textos de "query"
          (questões e exemplos sintéticos) com o prompt de instrução
          "Instruct: {tarefa}\nQuery:{texto}"; textos de "documento" (pseudo-documentos dos
          temas e descrições sintéticas) SEM prompt. Conjuntos: questoes, pseudo, sint, sint_desc.
  e5    = intfloat/multilingual-e5-base (usado SÓ por scripts/filtrar_sinteticos.py): prefixos
          "query: " e "passage: ". Conjuntos: pseudo, sint, sint_desc.

Entradas: questoes.jsonl, temas.csv, cache/sinteticos/ e prompts/temas_alvo.jsonl.
Saídas: <conjunto>.npy (float32, L2-normalizado) + <conjunto>_ids.json + manifest.json
(modelo, revisão, data, sha256) em cache/embeddings/<modelo>/.
Uso: python -B scripts/embeddings.py [--modelo qwen e5] [--force]   (baixa os pesos do Hugging Face se ainda não estiverem no cache)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessor import TextCleaner  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
EMB_DIR = BASE_DIR / "cache" / "embeddings"
MODELOS = {
    "qwen": dict(modelo="Qwen/Qwen3-Embedding-0.6B",
                 instrucao="Dada uma questao de prova juridica brasileira, recuperar o tema juridico (assunto) ao qual ela pertence",
                 max_len=512, batch=8, kw=dict(model_kwargs={"torch_dtype": "float32"}, tokenizer_kwargs={"padding_side": "left"}),
                 conjuntos=("questoes", "pseudo", "sint", "sint_desc")),
    "e5": dict(modelo="intfloat/multilingual-e5-base", instrucao=None, max_len=None, batch=16, kw={},
               conjuntos=("pseudo", "sint", "sint_desc")),
}


def sha256(p: Path) -> str:
    """Calcula o SHA-256 de um arquivo.

    Args:
        p: caminho do arquivo.

    Returns:
        Digest hexadecimal.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()


def textos_questoes():
    """Monta os textos normalizados das questões do case.

    Returns:
        Tupla (ids, textos, papel "query").
    """
    raw = [json.loads(l) for l in (BASE_DIR / "questoes.jsonl").read_text(encoding="utf-8").splitlines()]
    tx = [TextCleaner.extract_question_text(r.get("enunciado_html"), r.get("comando"), r.get("alternativas"), include_alternativas=True) for r in raw]
    return [r["id"] for r in raw], tx, "query"


def textos_pseudo():
    """Monta os pseudo-documentos dos temas de nível 1.

    Returns:
        Tupla (ids como texto, textos, papel "passage").
    """
    from models import HybridClassifier
    clf = HybridClassifier(BASE_DIR)
    return [str(t) for t in clf.tax.act_l1], list(clf.class_texts), "passage"


def carregar_sinteticos():
    """Carrega as questões e descrições sintéticas por tema-alvo.

    Um tema só é aceito com exatamente 1 descrição e 12 exemplos.

    Returns:
        Tupla (exemplos [(tema, índice, registro)], descrições [(tema, registro)]).

    Raises:
        SystemExit: se houver sintéticos incompletos.
    """
    alvo = [json.loads(l)["tema_id"] for l in (BASE_DIR / "prompts" / "temas_alvo.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ex, de, falt = [], [], []
    for t in sorted(alvo):
        p = BASE_DIR / "cache" / "sinteticos" / f"tema_{t}.jsonl"
        try:
            L = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        except (FileNotFoundError, json.JSONDecodeError):
            falt.append(t)
            continue
        d = [r for r in L if r.get("tipo") == "descricao"]
        e = [r for r in L if r.get("tipo") == "exemplo"]
        if len(d) != 1 or len(e) != 12:
            falt.append(t)
            continue
        de.append((t, d[0]))
        ex += [(t, i, r) for i, r in enumerate(e)]
    if falt:
        raise SystemExit(f"sinteticos incompletos: {len(falt)} temas faltando {falt}")
    return ex, de


def textos_sint():
    """Monta os textos dos exemplos sintéticos.

    Returns:
        Tupla (ids "tema#i", textos, papel "query").
    """
    ex, _ = carregar_sinteticos()
    return ([f"{t}#{i}" for t, i, _ in ex],
            [TextCleaner.extract_question_text(r.get("enunciado"), r.get("comando"), r.get("alternativas"), include_alternativas=True) for _, _, r in ex], "query")


def textos_sint_desc():
    """Monta os textos das descrições sintéticas (termos + caminho do tema).

    Returns:
        Tupla (ids "tema#desc", textos, papel "passage").
    """
    _, de = carregar_sinteticos()
    temas = pd.read_csv(BASE_DIR / "temas.csv").set_index("tema_id")
    tx = [" ".join([str(r.get("inclui", "")), " ".join(map(str, r.get("termos", []))), ""]).strip() + " " + str(temas.loc[int(t), "caminho"]) for t, r in de]
    return [f"{t}#desc" for t, _ in de], tx, "passage"


FONTES = {"questoes": textos_questoes, "pseudo": textos_pseudo, "sint": textos_sint, "sint_desc": textos_sint_desc}


def gerar(slug: str, force: bool) -> None:
    """Gera os conjuntos de embeddings de um modelo, na ordem e com os ids de FONTES.

    Args:
        slug: chave de MODELOS ("qwen" ou "e5").
        force: se True, refaz arquivos já existentes.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    cfg = MODELOS[slug]
    d = EMB_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = SentenceTransformer(cfg["modelo"], device="cpu", **cfg["kw"])
    model.eval()
    if cfg["max_len"]:
        model.max_seq_length = cfg["max_len"]
    try:
        from huggingface_hub import snapshot_download
        rev = Path(snapshot_download(cfg["modelo"])).name
    except Exception:
        rev = "desconhecida"
    mf = d / "manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {"arquivos": {}}
    man.update(modelo=cfg["modelo"], revisao=rev, max_seq_length=int(model.max_seq_length), data=datetime.date.today().isoformat(),
               normalizado=True, dtype="float32", instrucao_query=cfg["instrucao"])
    for nome in cfg["conjuntos"]:
        out = d / f"{nome}.npy"
        if out.exists() and not force:
            print(f"[skip] {slug}/{nome}")
            continue
        ids, tx, papel = FONTES[nome]()
        if slug == "e5":
            tx = [("query: " if papel == "query" else "passage: ") + t for t in tx]
        elif papel == "query":
            tx = [f"Instruct: {cfg['instrucao']}\nQuery:{t}" for t in tx]
        order = np.argsort([len(t) for t in tx]) if slug == "qwen" else np.arange(len(tx))  # qwen: ordena por tamanho só para acelerar o batch
        E = np.zeros((len(tx), model.get_sentence_embedding_dimension()), np.float32)
        E[order] = model.encode([tx[i] for i in order], batch_size=cfg["batch"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        np.save(out, E)
        (d / f"{nome}_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        man["arquivos"][nome] = dict(n=len(ids), dim=int(E.shape[1]), sha256=sha256(out), sha256_entrada=hashlib.sha256("\n".join(tx).encode()).hexdigest())
        print(f"[ok] {slug}/{nome}: {E.shape}", flush=True)
        mf.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    """Ponto de entrada: gera os embeddings dos modelos pedidos.

    Opções: --modelo (qwen, e5; padrão qwen) e --force.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", nargs="+", default=["qwen"], choices=list(MODELOS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for sl in a.modelo:
        gerar(sl, a.force)


if __name__ == "__main__":
    main()
