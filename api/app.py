"""API REST (FastAPI, com Swagger UI) para classificação de tópicos jurídicos.

Serviço opcional de demonstração; NÃO faz parte de ./replicar.sh (usa o
cache/modelo_producao.pkl que o replicar.sh grava). Expõe a mesma regra de
publicação (A1) do script de regeneração, lendo o pipeline ajustado.

Configuração entregue (A1): um roteador de raiz só de texto (LR sobre o embedding
Qwen3-Embedding-0.6B, sem disciplina/assunto) decide a área; Penal e Processo Penal
seguem a S4 (consenso NB=BM25=LSA + bagging + piso); Legislação Especial e
Criminologia seguem BM25 = qwen-LR + bagging + piso por área.

Separação exigida pelo case (seção 4): POST /ingerir faz o pré-processamento
(normalização, extração de leis e embedding Qwen3 local, só texto para vetor);
POST /classificar usa apenas o vetor guardado + o pipeline ajustado e NUNCA carrega
nem chama modelo de linguagem/codificador. Os vetores ingeridos ficam em memória e
em cache/ingestao/ (ignorado pelo git).
"""
from __future__ import annotations

import contextvars
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Adiciona scripts ao path
import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))

import calibracao as cb
import roteador_area as ra
from models import HybridClassifier
from preprocessor import TextCleaner
import pickle

app = FastAPI(
    title="API de Classificação de Tópicos Jurídicos (Case de classificação)",
    description=(
        "API de produção para classificação hierárquica automática de questões de concurso "
        "regra A1: roteador por área (qwen-LR de raiz) + S4 em Penal/Processo + consenso BM25=qwen-LR em Legislação Especial/Criminologia, com bagging e piso de score (ver previsao.txt). "
        "Utiliza Ensemble Híbrido Assimétrico (ComplementNB + Okapi BM25 + LSA + Senado + Fast-Track Normas), "
        "operando 100% offline. Fluxo em duas etapas: POST /ingerir (pré-processamento permitido pela seção 4: normalização, leis e embedding Qwen3 local, só texto para vetor) "
        "e POST /classificar (usa só o vetor guardado e o pipeline ajustado; nunca chama modelo de linguagem; sem vetor, responde 409)."
    ),
    version="2.0.0"
)

# Estado global da aplicação (Singleton em memória)
class ModelService:
    """Estado global da aplicação (singleton em memória).

    Attributes:
        classifier: HybridClassifier carregado no startup.
        modelo: dicionário do pipeline de produção (cache/modelo_producao.pkl).
        qwen: codificador Qwen3 (carregado sob demanda em /ingerir).
        leis: tupla (norma -> temas, nível, normas conhecidas) para extração de leis.
        store: registros de ingestão por id_ingestao.
    """
    classifier: HybridClassifier | None = None
    modelo = None
    qwen = None
    leis = None
    store: Dict[str, Dict[str, Any]] = {}

service = ModelService()


@app.on_event("startup")
def startup_event():
    """Carrega o pipeline de produção ajustado (cache/modelo_producao.pkl).

    O arquivo é escrito por scripts/gerar_publicacao.py. Também ajusta o LSA com as
    659 questões. Os codificadores só são carregados sob demanda.
    """
    classifier = HybridClassifier(BASE_DIR)
    raw_questions = [json.loads(line) for line in (BASE_DIR / "questoes.jsonl").read_text(encoding="utf-8").splitlines()]
    texts = [TextCleaner.extract_question_text(r.get("enunciado_html"), r.get("comando"), r.get("alternativas"), include_alternativas=True) for r in raw_questions]
    classifier.fit_lsa(texts)
    service.classifier = classifier
    service.modelo = pickle.loads((BASE_DIR / "cache" / "modelo_producao.pkl").read_bytes())
    # o encoder Qwen3 (get_qwen) so e carregado de forma preguicosa, em /ingerir


EM_CLASSIFICACAO = contextvars.ContextVar("em_classificacao", default=False)


def _proibir_codificador_na_classificacao():
    """Impede o uso de codificadores durante /classificar.

    Raises:
        RuntimeError: se chamada dentro do contexto de classificação.
    """
    if EM_CLASSIFICACAO.get():
        raise RuntimeError("codificador/modelo de linguagem proibido no caminho de /classificar (case, secao 4); use /ingerir")


INSTRUCAO_QWEN = ra.INSTRUCAO


def get_qwen():
    """Carrega o Qwen3-Embedding-0.6B local (mesmas opções de scripts/embeddings.py).

    Só é usado em /ingerir.

    Returns:
        SentenceTransformer do Qwen3-Embedding-0.6B.

    Raises:
        RuntimeError: se chamado durante /classificar.
    """
    _proibir_codificador_na_classificacao()
    if service.qwen is None:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu", local_files_only=True,
                                model_kwargs={"torch_dtype": "float32"}, tokenizer_kwargs={"padding_side": "left"})
        m.eval(); m.max_seq_length = 512
        service.qwen = m
    return service.qwen


def lei_da_questao(texto: str, act_l1: set):
    """Detecta a norma citada em um texto e o tema de nível 1 unívoco dela.

    Aplica a mesma regra de scripts/extrair_leis.py ao texto da requisição.

    Args:
        texto: texto da questão.
        act_l1: conjunto dos temas de nível 1 ativos.

    Returns:
        Tupla (citou alguma norma, tema de nível 1 unívoco ou None).
    """
    import extrair_leis as xl
    if service.leis is None:
        temas = pd.read_csv(BASE_DIR / "temas.csv"); lei_tema = pd.read_csv(BASE_DIR / "cache" / "lei_tema.csv")
        por_norma, pai, nivel = xl.carregar_mapa(lei_tema, temas)
        conhecidas = set(por_norma) | set(json.loads((BASE_DIR / "cache" / "normas_senado.json").read_text(encoding="utf-8"))["normas"])
        xl.PRIORIDADE.update(por_norma)
        service.leis = (por_norma, nivel, conhecidas)
    por_norma, nivel, conhecidas = service.leis
    normas = xl.normas_citadas(texto, conhecidas)
    n1 = sorted({tid for n in normas for tid in por_norma.get(n, []) if nivel.get(tid) == 1})
    return bool(normas), (n1[0] if len(n1) == 1 and n1[0] in act_l1 else None)


def classificar_area_nova(clf, M, R, texto, emb, rt, s_bm, s_ls):
    """Decide para Legislação Especial/Criminologia (A1).

    Regra: BM25 = qwen-LR no tema da raiz roteada (+ norma em Leg. Especial) + bagging
    + piso por área.

    Args:
        clf: HybridClassifier.
        M: dicionário do pipeline de produção.
        R: configuração de roteamento (M["rota"]).
        texto: texto da questão.
        emb: embedding Qwen3 da questão (1 x d).
        rt: id da raiz roteada.
        s_bm: scores BM25 normalizados.
        s_ls: scores LSA normalizados.

    Returns:
        Tupla (tema predito, consenso e norma ok, contagem de sacos, logit do blend,
        lista de motivos de revisão, título da área).
    """
    tax = clf.tax
    mod = R["mod"]; cols = mod["cols"]; K = len(tax.act_l1)
    def pt(m):
        """Calcula as probabilidades de tema expandidas a K colunas.

        Args:
            m: modelo de tema (LogisticRegression).

        Returns:
            Matriz (1 x K) de probabilidades.
        """
        P = np.zeros((1, K)); P[:, m.classes_] = m.predict_proba(emb)
        return P
    Pq = pt(mod["full"]); rts = np.array([rt])
    th = int(ra.restringir(cols, Pq, rts)[0]); bt = int(ra.restringir(cols, s_bm[None, :], rts)[0])
    cnt = sum(int(ra.restringir(cols, pt(mb), rts)[0] == th) for mb in mod["bags"])
    pred = tax.act_l1[th]
    cit, lei_tema = lei_da_questao(texto, set(tax.act_l1))
    law_ag = lei_tema == pred
    P7 = 0.55 * Pq[0] + 0.30 * s_bm + 0.15 * s_ls
    pc = float(np.clip(P7[th], 1e-6, 1 - 1e-6)); lg = float(np.log(pc / (1 - pc)))
    score = lg + 2.8 * law_ag
    gate = bt == th
    lawok = True if rt != R["areas_novas"][0] else ((law_ag and cit) if R["law_ok_leg"] == "cited" else (law_ag or not cit))
    motivos = []
    if not gate:
        motivos.append("sem consenso BM25 = qwen-LR na area roteada (" + tax.titulo[rt] + ")")
    else:
        if not lawok:
            motivos.append("a norma citada aponta outro tema")
        if R["need"] is not None and cnt < R["need"]:
            motivos.append(f"instavel sob bagging ({cnt}/{R['B']} sacos < {R['need']})")
        if score < mod["qa"][rt]:
            motivos.append("score do consenso abaixo do piso da area (quartil inferior)")
    return pred, gate and lawok, cnt, lg, motivos, tax.titulo[rt]


class QuestionInput(BaseModel):
    """Corpo de POST /ingerir.

    Attributes:
        enunciado_html: texto do enunciado com ou sem tags HTML.
        comando: comando final da questão.
        alternativas: dicionário de alternativas.
        disciplina_origem: disciplina informada na origem.
        assunto_origem: assunto informado na origem.
    """
    enunciado_html: str = Field(..., description="Texto do enunciado com ou sem tags HTML")
    comando: Optional[str] = Field(None, description="Comando final da questão")
    alternativas: Optional[Dict[str, str]] = Field(None, description="Dicionário de alternativas {'A': '...', 'B': '...'}")
    disciplina_origem: Optional[str] = Field("Direito Penal", description="Disciplina informada na origem")
    assunto_origem: Optional[str] = Field(None, description="Assunto informado na origem")


class ClassificarInput(BaseModel):
    """Corpo de POST /classificar.

    Attributes:
        id_ingestao: id devolvido por POST /ingerir (caminho recomendado).
        enunciado_html: questão completa (só vale se já foi ingerida).
        comando: comando final da questão.
        alternativas: dicionário de alternativas.
        disciplina_origem: disciplina informada na origem.
        assunto_origem: assunto informado na origem.
    """
    id_ingestao: Optional[str] = Field(None, description="Id devolvido por POST /ingerir (caminho recomendado).")
    enunciado_html: Optional[str] = Field(None, description="Questão completa (opcional): só vale se já foi ingerida; senão 409.")
    comando: Optional[str] = None
    alternativas: Optional[Dict[str, str]] = None
    disciplina_origem: Optional[str] = "Direito Penal"
    assunto_origem: Optional[str] = None


class IngestaoResponse(BaseModel):
    """Resposta de POST /ingerir.

    Attributes:
        id_ingestao: identificador do vetor guardado.
        nota: aviso de que nenhum tema é escolhido na ingestão.
    """
    id_ingestao: str = Field(..., description="Identificador do vetor guardado; use em POST /classificar")
    nota: str


NOTA_INGESTAO = "pré-processamento (permitido pela seção 4 do case); nenhum tema é escolhido aqui"
DIR_INGESTAO = BASE_DIR / "cache" / "ingestao"


def _id_ingestao(texto: str) -> str:
    """Calcula o identificador de ingestão de um texto.

    Args:
        texto: texto normalizado da questão.

    Returns:
        "ing_" seguido dos 16 primeiros caracteres hexadecimais do SHA-1.
    """
    return "ing_" + hashlib.sha1(texto.encode("utf-8")).hexdigest()[:16]


def _salvar(id_: str, reg: Dict[str, Any]):
    """Grava o registro de ingestão em cache/ingestao/<id>.npz.

    Args:
        id_: id da ingestão.
        reg: registro com texto, alternativas, metadados e vetor qwen.
    """
    DIR_INGESTAO.mkdir(parents=True, exist_ok=True)
    np.savez(DIR_INGESTAO / f"{id_}.npz", meta=np.array(json.dumps({k: reg[k] for k in ("texto", "alternativas", "disciplina_origem", "assunto_origem")}, ensure_ascii=False)),
             qwen=reg["qwen"])


def _carregar(id_: str):
    """Recupera um registro de ingestão, da memória ou do arquivo.

    Só lê vetores (sem modelo).

    Args:
        id_: id da ingestão.

    Returns:
        O registro, ou None se não existir ou o id for inválido.
    """
    if id_ in service.store:
        return service.store[id_]
    f = DIR_INGESTAO / f"{id_}.npz"
    if not f.exists() or not id_.startswith("ing_") or "/" in id_:
        return None
    z = np.load(f, allow_pickle=False)
    reg = json.loads(str(z["meta"])); reg["qwen"] = z["qwen"]
    service.store[id_] = reg
    return reg


class ClassificationResponse(BaseModel):
    """Resposta de POST /classificar.

    Attributes:
        decisao: "publicar" ou "revisar".
        tema_central_id: id do tema na árvore (temas.csv).
        tema_central_titulo: nome do tema central.
        tema_central_caminho: caminho hierárquico na taxonomia.
        confianca: probabilidade calibrada de o ramo do tema sugerido estar correto.
        motivo: justificativa da decisão.
    """
    decisao: str = Field(..., description="'publicar' ou 'revisar'")
    tema_central_id: int = Field(..., description="ID do tema na árvore (temas.csv)")
    tema_central_titulo: str = Field(..., description="Nome do tema central")
    tema_central_caminho: str = Field(..., description="Caminho hierárquico na taxonomia")
    confianca: float = Field(..., description="Probabilidade calibrada (OOF aninhado) de o ramo do tema sugerido estar correto")
    motivo: str = Field(..., description="Justificativa da decisão tomada")


class BatchClassificationResponse(BaseModel):
    """Resposta de classificação em lote.

    Attributes:
        total: número de questões.
        publicadas: quantas foram publicadas.
        revisar: quantas foram para revisão.
        questoes: respostas individuais.
    """
    total: int
    publicadas: int
    revisar: int
    questoes: List[ClassificationResponse]


@app.get("/", tags=["Health"])
def health_check():
    """Verifica se a API está online e pronta para inferência.

    Returns:
        Dicionário com status, modelo, indicação de operação sem LLM, fluxo e número
        de temas ativos.
    """
    return {
        "status": "healthy",
        "model": "A1: roteador por area (qwen-LR) + S4 em Penal/Processo + consenso BM25=qwen-LR em Leg. Especial/Criminologia",
        "offline_llm": True,
        "fluxo": "POST /ingerir (embedding local) e depois POST /classificar (sem modelo de linguagem)",
        "taxonomia_nós_ativos": len(service.classifier.tax.act_l1) if service.classifier else 0
    }


@app.post("/ingerir", response_model=IngestaoResponse, tags=["Ingestão (pré-processamento)"])
def ingerir_questao(q: QuestionInput):
    """Pré-processa e guarda o vetor de uma questão (etapa permitida pela seção 4 do case).

    Normaliza o texto, extrai as leis citadas e gera LOCALMENTE (local_files_only=True)
    o embedding Qwen3-Embedding-0.6B, que apenas converte texto em vetor. O vetor fica
    na memória do processo e em cache/ingestao/. Nenhum tema é escolhido aqui e nenhum
    texto é gerado; a decisão fica em POST /classificar.

    Args:
        q: questão a ingerir.

    Returns:
        IngestaoResponse com o id da ingestão.

    Raises:
        HTTPException: 500 se o modelo não estiver carregado.
    """
    if service.classifier is None or service.modelo is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    texto = TextCleaner.extract_question_text(q.enunciado_html, q.comando, q.alternativas, include_alternativas=True)
    id_ = _id_ingestao(texto)
    if _carregar(id_) is None:
        qwen = get_qwen().encode([f"Instruct: {INSTRUCAO_QWEN}\nQuery:{texto}"], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        lei_da_questao(texto, set(service.classifier.tax.act_l1))  # aquece o mapa de leis (regex, sem modelo)
        reg = {"texto": texto, "alternativas": q.alternativas, "disciplina_origem": q.disciplina_origem, "assunto_origem": q.assunto_origem, "qwen": qwen}
        service.store[id_] = reg
        _salvar(id_, reg)
    return IngestaoResponse(id_ingestao=id_, nota=NOTA_INGESTAO)


@app.post("/classificar", response_model=ClassificationResponse, tags=["Classificação"])
def classificar_questao(q: ClassificarInput):
    """Classifica uma questão JÁ INGERIDA (id_ingestao de POST /ingerir).

    Usa só o vetor guardado e o pipeline de cache/modelo_producao.pkl: não carrega nem
    chama modelo de linguagem ou codificador. Se vier a questão completa sem vetor
    guardado, responde 409. O roteador de raiz (só texto) escolhe a área; Penal e
    Processo publicam só se NB=BM25=LSA concordam, estável sob bagging e acima do piso;
    Legislação Especial e Criminologia usam BM25 = qwen-LR.

    Args:
        q: id_ingestao ou questão completa já ingerida.

    Returns:
        ClassificationResponse com decisão, tema, confiança e motivo.

    Raises:
        HTTPException: 500 (modelo não carregado), 422 (nem id nem questão) ou 409
            (sem vetor guardado).
    """
    if service.classifier is None or service.modelo is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    id_ = q.id_ingestao
    if id_ is None and q.enunciado_html is not None:
        texto = TextCleaner.extract_question_text(q.enunciado_html, q.comando, q.alternativas, include_alternativas=True)
        id_ = _id_ingestao(texto)
    if id_ is None:
        raise HTTPException(status_code=422, detail="Envie id_ingestao (de POST /ingerir) ou a questão completa já ingerida.")
    reg = _carregar(id_)
    if reg is None:
        raise HTTPException(status_code=409, detail="Não há vetor guardado para esta questão: chame POST /ingerir primeiro (a classificação não gera embeddings).")
    token = EM_CLASSIFICACAO.set(True)
    try:
        return _decidir(reg)
    finally:
        EM_CLASSIFICACAO.reset(token)


def _decidir(reg: Dict[str, Any]) -> ClassificationResponse:
    """Calcula a decisão a partir do vetor guardado; nenhum codificador é usado.

    Verificado em tests/test_api_ingestao.py. O roteador de raiz (A1) escolhe a área;
    em Legislação Especial e Criminologia vale BM25 = qwen-LR; em Penal e Processo, o
    consenso NB=BM25=LSA com bagging e piso de score. Calcula a confiança calibrada.

    Args:
        reg: registro de ingestão (texto, alternativas, metadados e vetor).

    Returns:
        ClassificationResponse com a decisão.
    """
    clf, M = service.classifier, service.modelo
    tax = clf.tax
    texto = reg["texto"]
    s_bm = clf.compute_bm25_matrix([texto])[0]; s_bm = s_bm / max(s_bm.sum(), 1e-9)
    s_ls = clf.compute_lsa_matrix([texto])[0]; s_ls = s_ls / max(s_ls.sum(), 1e-9)
    one = pd.DataFrame([{"id": "API", "texto_modelo": texto, "disciplina_origem": reg["disciplina_origem"], "assunto_origem": reg["assunto_origem"]}])
    R = M["rota"]
    emb = reg["qwen"]
    rt = int(R["mod"]["router"].classes_[int(np.argmax(R["mod"]["router"].predict_proba(emb)[0]))])
    if rt in R["areas_novas"]:
        pred, cons, cnt, lg, motivos, area = classificar_area_nova(clf, M, R, texto, emb, rt, s_bm, s_ls)
        publicar = cons and not motivos
        confianca = round(float(cb.aplicar(M["cal"], cb.feats([publicar], [cons], [cnt], [lg]))[0]), 4)
        motivo = (f"consenso BM25 = qwen-LR estavel na area {area} ({M['nome'].split()[0]}; precisao honesta e cobertura em previsao.txt)" if publicar
                  else "revisar: " + "; ".join(motivos) + "; sugestao encaminhada para revisao")
        return ClassificationResponse(decisao="publicar" if publicar else "revisar", tema_central_id=pred,
                                      tema_central_titulo=tax.titulo.get(pred, ""), tema_central_caminho=tax.caminho.get(pred, ""),
                                      confianca=confianca, motivo=motivo)
    p_nb = clf.predict_hierarchy(M["full"], one)[0]
    p_bl = 0.55 * p_nb + 0.30 * s_bm + 0.15 * s_ls
    b = int(np.argmax(p_bl))
    pred = tax.act_l1[b]
    consenso = bool(p_nb.sum() > 0 and int(np.argmax(p_nb)) == b and int(np.argmax(s_bm)) == b and int(np.argmax(s_ls)) == b)
    cnt = sum(int(P.sum() > 0 and int(np.argmax(P)) == b) for P in (clf.predict_hierarchy(st, one)[0] for st in M["bags"]))
    pc = float(np.clip(p_bl[b], 1e-6, 1 - 1e-6))
    score = float(np.log(pc / (1 - pc)))
    motivos = []
    if not consenso:
        motivos.append("sem consenso NB=BM25=LSA")
    else:
        if cnt < M["need"]:
            motivos.append(f"instavel sob bagging ({cnt}/{M['B']} sacos < {M['need']})")
        if score < M["qg"]:
            motivos.append("score do consenso abaixo do piso (quartil inferior)")
    publicar = consenso and not motivos
    confianca = round(float(cb.aplicar(M["cal"], cb.feats([publicar], [consenso], [cnt], [score]))[0]), 4)
    motivo = (f"consenso NB=BM25=LSA estavel e acima do piso ({M['nome'].split()[0]}; precisao honesta e cobertura em previsao.txt)" if publicar
              else "revisar: " + "; ".join(motivos) + "; sugestao encaminhada para revisao")
    return ClassificationResponse(decisao="publicar" if publicar else "revisar", tema_central_id=pred,
                                  tema_central_titulo=tax.titulo.get(pred, ""), tema_central_caminho=tax.caminho.get(pred, ""),
                                  confianca=confianca, motivo=motivo)


@app.get("/temas", tags=["Taxonomia"])
def listar_temas():
    """Lista todos os temas ativos de nível 1 aceitos para classificação.

    Returns:
        Lista de dicionários com tema_id, titulo e caminho.

    Raises:
        HTTPException: 500 se o modelo não estiver carregado.
    """
    if service.classifier is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    tax = service.classifier.tax
    return [
        {"tema_id": tid, "titulo": tax.titulo[tid], "caminho": tax.caminho[tid]}
        for tid in tax.act_l1
    ]
