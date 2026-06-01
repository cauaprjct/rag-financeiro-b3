# Textos de divulgação (GitHub + LinkedIn)

> Arquivo auxiliar. Pode apagar antes do `git push` se não quiser no repositório.

---

## 1) GitHub — "About" (descrição curta do repo)

```
RAG sobre documentos financeiros da B3: PDF -> respostas fundamentadas com citações. Retrieval híbrido (denso + BM25 + RRF), reranking opcional, avaliação LLM-as-judge, 27 testes property-based e CI. Python 3.12 + Streamlit.
```

## 2) GitHub — Topics (tags)

```
rag  retrieval-augmented-generation  llm  python  streamlit  chromadb
sentence-transformers  bm25  reranking  gemini  groq  hypothesis  property-based-testing
```

## 3) GitHub — primeiro commit sugerido

```
feat: agente RAG financeiro sobre documentos da B3

Pipeline em camadas (ingestao -> chunking -> embeddings -> vector store ->
retrieval hibrido + reranking opcional -> geracao com citacoes -> avaliacao).
89 testes (27 property-based com hypothesis), CI e execucao offline com fakes.
```

---

## 4) LinkedIn — post (PT-BR)

Construí um agente de **RAG sobre documentos financeiros da B3**: você sobe os PDFs (releases, demonstrações) e pergunta em linguagem natural — a resposta vem **fundamentada nos trechos**, com **citação de documento e página**.

O que eu priorizei foi **engenharia**, não só "chamar um LLM":

🔹 **Retrieval híbrido** — busca densa (embeddings multilingual-e5) + BM25, fundidos com Reciprocal Rank Fusion. Melhor recall em termos exatos (tickers, números) e semânticos.

🔹 **Reranking opcional, desligado por padrão** — cross-encoder melhora a precisão, mas pesa em CPU. Decisão consciente: deploy gratuito roda leve; a GPU é aproveitada só no dev. Se o reranker falhar, cai para o retrieval base sem quebrar a consulta.

🔹 **Avaliação automatizada** — módulo LLM-as-judge (Context Precision, Faithfulness) + Response Relevancy por similaridade. Dá pra medir regressão, não só "parece bom".

🔹 **Qualidade de código** — 89 testes, sendo **27 property-based (hypothesis)**, CI no GitHub Actions e toda a suíte roda **offline** com fakes determinísticos (sem rede, sem custo de API).

Stack: Python 3.12, Streamlit, ChromaDB, sentence-transformers, Groq/Gemini.

A parte mais valiosa não foi o "feliz caminho", e sim os **tradeoffs**: portabilidade x precisão, custo x qualidade, e tratar o LLM como dependência falível (timeout, rate limit) em vez de mágica.

Código aberto 👉 [LINK DO REPO]

#RAG #LLM #Python #MachineLearning #IA #DataScience #Engenharia

---

## 5) LinkedIn — versão curta (alternativa)

Cansei de RAG que é só "embeddings + 1 chamada de LLM". Então construí um **agente RAG sobre documentos da B3** levando a sério a engenharia:

• Retrieval híbrido (denso + BM25 + RRF) com citações de origem
• Reranking opcional com fallback gracioso (decisão de portabilidade)
• Avaliação LLM-as-judge para medir qualidade
• 89 testes (27 property-based) + CI, rodando offline com fakes

Python 3.12 · Streamlit · ChromaDB · sentence-transformers · Groq/Gemini

👉 [LINK DO REPO]

#RAG #LLM #Python #IA

---

## Checklist antes de publicar

- [ ] `git init`, commit e push para o GitHub
- [ ] Preencher `.env` (ou `.streamlit/secrets.toml`) com chave **Groq** ou **Gemini** válida
- [ ] Testar localmente: `streamlit run app.py`
- [ ] (Opcional) Deploy no Streamlit Community Cloud e colar o link no README
- [ ] Adicionar 1 screenshot/GIF no topo do README
- [ ] Trocar `[LINK DO REPO]` pelos links reais
