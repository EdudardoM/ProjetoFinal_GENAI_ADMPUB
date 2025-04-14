#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Carregar variáveis do .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Constantes
PDF_PATH = "recorte_manual_emenda.pdf"
PERSIST_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
N_DOCUMENTOS = 3

# Título do app
st.set_page_config(page_title="RAG Público 📘🤖", page_icon="📘")
st.title("Chat com o Manual de Emendas 📘")
st.markdown("**Trabalho Final Disciplina IAGEN Adm Pública** - MBA CDIA Enap")
st.markdown("***Eduardo Moura***")

# Carregar ou criar vetores (cache para performance)
@st.cache_resource
def carregar_ou_criar_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIR):
        return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)

    loader = PyPDFLoader(PDF_PATH)
    dados = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    textos = splitter.split_documents(dados)

    vectordb = FAISS.from_documents(textos, embeddings)
    vectordb.save_local(PERSIST_DIR)
    return vectordb

# Carrega o vetor
vector_db = carregar_ou_criar_vectordb()

# Inicializar modelo
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
prompt_template = hub.pull("rlm/rag-prompt")

def format_docs(documentos):
    return "\n\n".join(doc.page_content for doc in documentos)

rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": vector_db.as_retriever(k=N_DOCUMENTOS) | format_docs,
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# Inicializar sessão de histórico de chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Exibir histórico
for mensagem in st.session_state.chat_history:
    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])

# Entrada do usuário (chat)
pergunta = st.chat_input("Digite sua pergunta sobre o manual...")

if pergunta:
    # Adiciona a pergunta ao histórico
    st.session_state.chat_history.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Gera resposta
    with st.chat_message("assistant"):
        st.info("Pensando 🤔...")
        resposta = rag_chain.invoke(pergunta)
        st.success("Resposta:")
        st.markdown(resposta)
    
    # Adiciona a resposta ao histórico
    st.session_state.chat_history.append({"role": "assistant", "content": resposta})

