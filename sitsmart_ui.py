import os
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from transformers import pipeline
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from secrets or environment variables
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

client = Groq(api_key=api_key)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

prompt_template = """You are an intelligent Question Answer system that can answer questions from the catalog of a chairs, which is provided as the context.
Answer the question based on the context below, and if the question can't be answered based on the context, say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {input}
Context: {context}
Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["input", "context"])

@st.cache_resource
def data_ingestion(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        with open(os.path.join("documents", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFDirectoryLoader("documents")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs.extend(text_splitter.split_documents(data))
    return docs

@st.cache_resource
def vector_embeddings(docs):
    with st.sidebar:
        st.write("Embeddings...")
        db = FAISS.from_documents(docs, embeddings)
        st.write("Indexing...")
        db.save_local("faiss_index1")
        st.write("Indexing Done...")
    return db

def groq_model():
    chat_model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", api_key=os.environ.get("GROQ_API_KEY"))
    return chat_model

def get_retrieval_qa(llm, context, query):
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    retrieval_chain = create_retrieval_chain(None, document_chain)  # No retriever, direct context use
    response = retrieval_chain.invoke({"input": query, "context": context})
    return response

@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query, docs, reranker):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    sorted_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in sorted_docs]

st.title("Sitsmart Search")

# Sidebar for upload and update vector DB
with st.sidebar:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload documents", type=["pdf"], accept_multiple_files=True)
    update_button = st.button("Update Vector DB Index")

# Main area for query input and results
query = st.text_input("Enter your query:", placeholder="Type your query here...")

if update_button:
    with os.scandir("documents") as it:
        if not any(it):
            st.warning(" NO documents found Please upload documents first.")
    else:
        if not os.path.exists("documents"):
            os.makedirs("documents")
        docs = data_ingestion(uploaded_files)
        db = vector_embeddings(docs)
        st.success("Vector DB Index updated successfully.")

if query:
    llm = groq_model()
    try:
        faiss_index = FAISS.load_local("faiss_index1", embeddings, allow_dangerous_deserialization=True)
    except:
        st.warning("Please upload documents first or refresh db.")
        faiss_index = None

    if faiss_index:
        # Retrieve initial documents
        retrieved_docs = faiss_index.similarity_search(query, k=10)

        # Re-rank the retrieved documents
        reranker = load_reranker()
        sorted_docs = rerank_documents(query, retrieved_docs, reranker)

        # Use only the top N documents for the final answer
        top_docs = sorted_docs[:3]
        context = " ".join([doc.page_content for doc in top_docs])

        # Generate the final answer
        response = get_retrieval_qa(llm, context, query)
        st.write(response['answer'])

        # Display source documents in a scrollable window
        with st.expander("Source Documents"):
            with st.container():
                for doc in top_docs:
                    st.write(f"- {doc.metadata['source']}, Page {doc.metadata['page']}")
                    st.write(doc.page_content)
                    st.write("---")
