# sqlite workaround for Chroma on Streamlit Cloud (must run before chromadb import)
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import tempfile
from pathlib import Path

import streamlit as st

from core.config import load_config, ConfigError
from core.pipeline import build_pipeline

DISCLAIMER = (
    "⚠️ As respostas são geradas por IA a partir dos documentos carregados e "
    "podem conter erros. Não constituem recomendação de investimento."
)

st.set_page_config(page_title="RAG Financeiro B3", page_icon="📊")


def get_pipeline():
    if "pipeline" not in st.session_state:
        cfg = load_config(dict(st.secrets) if hasattr(st, "secrets") else None)
        st.session_state.pipeline = build_pipeline(cfg)
    return st.session_state.pipeline


def main():
    st.title("📊 RAG Financeiro sobre Documentos da B3")
    st.caption(DISCLAIMER)  # 10.1: permanent disclaimer

    if "history" not in st.session_state:
        st.session_state.history = []

    try:
        pipeline = get_pipeline()
    except ConfigError as e:
        st.error(f"Erro de configuração: {e}")
        return

    # --- Sidebar: upload and document management ---
    with st.sidebar:
        st.header("Documentos")
        uploaded = st.file_uploader("Envie 1 a 20 PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded and st.button("Indexar"):
            if len(uploaded) > 20:  # 8.1
                st.error("Máximo de 20 arquivos por vez.")
            else:
                _index_uploaded(pipeline, uploaded)

        docs = pipeline.store.list_documents()
        if not docs:  # 8.5
            st.info("Nenhum documento carregado.")
        else:
            st.metric("Documentos", len(docs))  # 8.10
            st.metric("Trechos", pipeline.store.count())
            for d in docs:
                col1, col2 = st.columns([3, 1])
                col1.write(f"{d.name} ({d.chunk_count})")
                if col2.button("🗑", key=f"del_{d.name}"):  # 8.9
                    pipeline.store.delete_document(d.name)
                    st.rerun()  # 8.11: refresh list

    # --- Document filter (8.13, 8.14) ---
    doc_names = [d.name for d in pipeline.store.list_documents()]
    selected = st.multiselect("Filtrar por documentos (vazio = todos)", doc_names)
    document_filter = selected or None  # 8.14: empty -> all

    # --- Query ---
    question = st.text_input("Sua pergunta", max_chars=1000)  # 8.2
    if st.button("Perguntar"):
        if not question.strip():  # 8.6
            st.warning("Digite uma pergunta.")
        else:
            _answer(pipeline, question, document_filter)

    # --- History (8.7, 8.8) ---
    for item in reversed(st.session_state.history):
        st.subheader(f"❓ {item['question']}")
        # Escape '$' to avoid Streamlit interpreting "R$ 1,75" as LaTeX math
        st.markdown(item["answer"].replace("$", r"\$"))
        st.caption(DISCLAIMER)  # 10.2: disclaimer with answer
        for c in item["citations"]:
            snippet = c.snippet.replace("$", r"\$")
            st.markdown(f"- **{c.document_name}** (p. {c.page_number}): {snippet}")  # 8.3


def _index_uploaded(pipeline, uploaded):
    with st.spinner("Indexando documentos..."):  # 8.4
        paths = []
        tmpdir = Path(tempfile.mkdtemp())
        for f in uploaded:
            p = tmpdir / f.name
            p.write_bytes(f.getbuffer())
            paths.append(p)
        results = pipeline.index_files(paths)
    for r in results:
        if r.success:
            st.success(f"{r.document_name} indexado.")
        else:
            st.error(f"{r.document_name}: {r.error}")


def _answer(pipeline, question, document_filter):
    with st.spinner("Consultando..."):  # 8.4
        result = pipeline.query(question, document_filter=document_filter)
    if result.status in ("unavailable", "rate_limited"):  # 8.12: error preserves session
        st.error(result.answer)
    st.session_state.history.append(
        {
            "question": question,
            "answer": result.answer,
            "citations": result.citations,
        }
    )


if __name__ == "__main__":
    main()
