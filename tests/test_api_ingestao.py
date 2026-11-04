"""Testes da separação /ingerir (embedding local) e /classificar (sem modelo de linguagem).

Não faz parte de ./replicar.sh; valida a API (api/app.py) contra publicacao.csv.
Rodar: ../venv/bin/python -B -m pytest -p no:cacheprovider tests
Entradas: questoes.jsonl, publicacao.csv e cache/modelo_producao.pkl. As ingestões
dos testes vão para um diretório temporário (não deixam cache/ingestao).
"""
import csv, inspect, json, sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.dont_write_bytecode = True
import api.app as A

QUESTOES = {}
for l in (BASE / "questoes.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(l); QUESTOES[r["id"]] = r
PUB = {r["id"]: r for r in csv.DictReader(open(BASE / "publicacao.csv", encoding="utf-8"))}


def corpo(i):
    """Monta o corpo JSON de uma requisição a partir de uma questão do case.

    Args:
        i: id da questão (ex.: "Q0001").

    Returns:
        Dicionário com enunciado_html, comando, alternativas, disciplina_origem e assunto_origem.
    """
    r = QUESTOES[i]
    return {k: r.get(k) for k in ("enunciado_html", "comando", "alternativas", "disciplina_origem", "assunto_origem")}




@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    """Fixture de módulo: cliente de teste da API com ingestão em diretório temporário.

    Args:
        tmp_path_factory: fábrica de diretórios temporários do pytest.

    Yields:
        TestClient com a aplicação iniciada (evento startup executado).
    """
    A.DIR_INGESTAO = tmp_path_factory.mktemp("ingestao")
    with TestClient(A.app) as c:
        yield c


def escolher(cli):
    """Escolhe questões de amostra de cada área/decisão a partir de publicacao.csv.

    Seleciona a primeira questão de Penal publicada, Legislação Especial publicada e
    revisar, e Criminologia publicada.

    Args:
        cli: cliente de teste da API (para consultar /temas).

    Returns:
        Lista com os ids encontrados.
    """
    temas = {t["tema_id"]: t["caminho"] for t in cli.get("/temas").json()}
    def a(r):
        """Classifica a área de uma linha de publicacao.csv.

        Args:
            r: linha do CSV com tema_central_id.

        Returns:
            "esp" (Legislação Especial), "crim" (Criminologia) ou "penal" (demais).
        """
        c = temas.get(int(r["tema_central_id"]), "")
        return "esp" if "Especial" in c else "crim" if "Criminologia" in c else "penal"
    alvo = {("penal", "publicar"): None, ("esp", "publicar"): None, ("esp", "revisar"): None, ("crim", "publicar"): None}
    for i, r in PUB.items():
        k = (a(r), r["decisao"])
        if k in alvo and alvo[k] is None:
            alvo[k] = i
    return [v for v in alvo.values() if v]


def test_409_sem_ingestao(cli):
    """Verifica que /classificar sem ingestão prévia responde 409 pedindo /ingerir.

    Args:
        cli: cliente de teste da API.
    """
    r = cli.post("/classificar", json=corpo("Q0001"))
    assert r.status_code == 409 and "/ingerir" in r.json()["detail"]
    assert cli.post("/classificar", json={"id_ingestao": "ing_inexistente"}).status_code == 409


def test_codigo_de_classificar_nao_toca_codificador():
    """Verifica no código-fonte que o caminho de /classificar não usa codificador."""
    for fn in (A.classificar_questao, A._decidir):
        src = inspect.getsource(fn)
        assert "SentenceTransformer" not in src and "get_qwen" not in src and ".encode(" not in src


def test_ingerir_e_classificar_batem_com_publicacao_csv(cli, monkeypatch):
    """Verifica que ingerir + classificar reproduz as decisões de publicacao.csv.

    Também garante que nenhum codificador é carregado ou chamado durante /classificar.

    Args:
        cli: cliente de teste da API.
        monkeypatch: fixture do pytest para trocar os codificadores por uma função que falha.
    """
    ids = escolher(cli)
    assert len(ids) == 4, ids
    ing = {}
    for i in ids:
        r = cli.post("/ingerir", json=corpo(i)); assert r.status_code == 200
        assert "nenhum tema é escolhido aqui" in r.json()["nota"]
        ing[i] = r.json()["id_ingestao"]
    # (c) a partir daqui qualquer uso do codificador falha o teste
    def proibido(*a, **k):
        """Substituto que falha o teste se um codificador for chamado.

        Args:
            *a: argumentos posicionais ignorados.
            **k: argumentos nomeados ignorados.

        Raises:
            AssertionError: sempre.
        """
        raise AssertionError("codificador chamado em /classificar")
    monkeypatch.setattr(A, "get_qwen", proibido)
    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", proibido)
    A.service.qwen = None
    for i in ids:
        r = cli.post("/classificar", json={"id_ingestao": ing[i]}); assert r.status_code == 200, r.text
        j = r.json(); esp = PUB[i]
        assert j["decisao"] == esp["decisao"] and str(j["tema_central_id"]) == esp["tema_central_id"], (i, j, esp)
        # questao completa ja ingerida tambem funciona
        assert cli.post("/classificar", json=corpo(i)).json()["decisao"] == esp["decisao"]
    assert A.service.qwen is None


def test_guarda_em_tempo_de_execucao():
    """Verifica que get_qwen levanta RuntimeError dentro do contexto de classificação."""
    tok = A.EM_CLASSIFICACAO.set(True)
    try:
        with pytest.raises(RuntimeError):
            A.get_qwen()
    finally:
        A.EM_CLASSIFICACAO.reset(tok)
