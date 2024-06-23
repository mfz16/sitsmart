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
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import torch
from transformers import pipeline
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

#@st.cache_resource
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

# def gemma_llm_local():
#         # Define quantization configuration
#         quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         )

#         # Define the model path
#         model_path = "D:\mymodels\gemma-2b-it_local"


#         # Load the tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(
#         model_path, quantization_config=quantization_config, torch_dtype=torch.float16
#         )

#         # Load the model
#         model = AutoModelForCausalLM.from_pretrained(
#         model_path, quantization_config=quantization_config, torch_dtype=torch.float16
#         )

#         # Define text generation pipeline
#         generation_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         max_new_tokens=512,
#         )

#         # Define LLM pipeline
#         gemma_llm = HuggingFacePipeline(
#         pipeline=generation_pipeline,
#         model_kwargs={"temperature": 0.7},
#         )
#         return gemma_llm

def get_retrieval_qa(llm, db, query):
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    retriever = db.as_retriever(search_kwargs={"k": 10})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
    response = retrieval_chain.invoke({"input": query})
    return response

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
    #if not uploaded_files:
        #st.warning("Please upload documents first.")
        else:
            if not os.path.exists("documents"):
                os.makedirs("documents")
            docs = data_ingestion(uploaded_files)
            db = vector_embeddings(docs)
            st.success("Vector DB Index updated successfully.")


if query:
    llm = groq_model()
    #llm= gemma_llm_local()
    try:
        faiss_index = FAISS.load_local("faiss_index1", embeddings, allow_dangerous_deserialization=True)
    except:
        st.warning("Please upload documents first or refresh db.")
        faiss_index=None
    if faiss_index:
        answer = get_retrieval_qa(llm, faiss_index, query)
        st.write(answer['answer'])

        # Display source documents in a scrollable window
        with st.expander("Source Documents"):
            with st.container():
                    for ans in answer['context']:
                        st.write(f"- {ans.metadata['source']}, Page {ans.metadata['page']}")
                        st.write(ans.page_content)
                        st.write("---")