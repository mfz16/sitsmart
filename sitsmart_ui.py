import os
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API key
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Prompt template
prompt_template = """You are an intelligent Question Answer system that can answer questions from the catalog of chairs from a company named Sitsmart,
which is provided as the context. Answer the question based on the context below, and if the question can't be answered based on the context, 
say "Hmm, I'm not sure." Don't try to make up an answer.

Question: {input}
Context: {context}
Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["input", "context"])

# Document ingestion
@st.cache_resource
def data_ingestion(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        if not os.path.exists("documents"):
            os.makedirs("documents")
        with open(os.path.join("documents", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    loader = PyPDFDirectoryLoader("documents")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs.extend(text_splitter.split_documents(data))
    return docs

# Vector DB creation
def vector_embeddings(docs):
    with st.sidebar:
        st.write("Embedding and Indexing...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index1")
    with st.sidebar:
        st.success("Indexing Done.")
    return db

# Load LLM
def groq_model():
    return ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",  # or llama-3-8b-8192
        api_key=os.environ.get("GROQ_API_KEY")
    )

# Retrieval-based QA
def get_retrieval_qa(llm, db, query):
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
    return retrieval_chain.invoke({"input": query})


# -------------------------------
# 🧠 Streamlit UI starts here
# -------------------------------

st.set_page_config(page_title="Sitsmart Chat", layout="wide")
st.title("🪑 Sitsmart Assistant - How Can I help?")

# Sidebar for uploading files and updating index
with st.sidebar:
    st.subheader("📄 Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("🔄 Update Vector DB Index"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            docs = data_ingestion(uploaded_files)
            vector_embeddings(docs)

# Initialize session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something about chairs..."):
    # Show user's message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Load model and index
    llm = groq_model()
    try:
        db = FAISS.load_local("faiss_index1", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        db = None
        st.warning("No index found. Please upload and index documents first.")

    if db:
        # Run retrieval QA
        result = get_retrieval_qa(llm, db, prompt)
        answer = result['answer']

        # Show assistant's message
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Show source documents
        with st.expander("📚 Source Documents"):
            for doc in result['context']:
                st.markdown(f"**{doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', '?')}**")
                st.markdown(doc.page_content)
                st.markdown("---")
