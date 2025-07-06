"""Document processing utilities for SitSmart application"""
import os
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from .image_processing import extract_images_from_pdfs, clear_old_images

# Initialize embeddings with caching
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load LLM with caching
@st.cache_resource
def groq_model():
    return ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        api_key=os.environ.get("GROQ_API_KEY")
    )

# Cache reranker model
@st.cache_resource
def get_reranker():
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    return CrossEncoderReranker(model=model, top_n=2)

# Cache document chain
@st.cache_resource
def get_document_chain(_llm):
    prompt_template = """You are SitSmart Assistant for SitSmart India's chair catalog.

**Instructions:**
- Provide specific product details, prices, and features from the context
- Use bullet points for specifications
- If info missing, say "Contact our sales team for details"
- Be helpful and professional

**Context:** {context}
**Question:** {input}
**Answer:**"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["input", "context"])
    return create_stuff_documents_chain(_llm, PROMPT)

def extract_product_name(doc_content):
    """Extract Model No. from table: empty | Model | Model No. | Description | Price"""
    lines = doc_content.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        if 'model no' in line.lower() or 'model' in line.lower():
            for j in range(i+1, min(i+5, len(lines))):
                data_line = lines[j].strip()
                if not data_line:
                    continue
                
                parts = re.split(r'\s{2,}|\t|\|', data_line)
                
                if len(parts) >= 3:
                    model_no = parts[2].strip()
                    if re.match(r'^[A-Z]{2,4}\s*-\s*[A-Z0-9]{2,4}$', model_no, re.IGNORECASE):
                        return model_no.upper()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        ssc_patterns = [
            r'SSC\s*-\s*[A-Z0-9]{2,4}',
            r'SSC\s+[A-Z0-9]{2,4}',
        ]
        
        for pattern in ssc_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group().upper().replace(' ', ' ')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        words = line.split()
        for word in words:
            if re.match(r'^[A-Z]{2,4}-[A-Z0-9]{2,4}$', word, re.IGNORECASE):
                return word.upper()
    
    return "Product"

def get_relevant_images_with_names(doc_metadata, doc_content):
    """Get images related to specific products on a page"""
    source = doc_metadata.get('source', '')
    page = doc_metadata.get('page', 0)
    
    if not os.path.exists("images"):
        return []
    
    filename = os.path.basename(source)
    product_name = extract_product_name(doc_content)
    
    if product_name == "Product":
        return []
    
    images_with_names = []
    product_key = product_name.replace(' ', '_').replace('-', '_')
    
    for img_file in os.listdir("images"):
        if (img_file.startswith(filename) and 
            f"page{page}" in img_file and 
            product_key in img_file):
            img_path = os.path.join("images", img_file)
            images_with_names.append((img_path, product_name))
    
    if not images_with_names:
        for img_file in os.listdir("images"):
            if img_file.startswith(filename) and f"page{page}" in img_file:
                img_path = os.path.join("images", img_file)
                images_with_names.append((img_path, product_name))
    
    return images_with_names[:1]

def filter_images_by_answer(images_with_names, answer_text):
    """Only return images for products mentioned in the LLM answer"""
    if not images_with_names or not answer_text:
        return []
    
    filtered_images = []
    answer_upper = answer_text.upper()
    
    for img_path, product_name in images_with_names:
        product_upper = product_name.upper()
        
        if product_upper in answer_upper:
            filtered_images.append((img_path, product_name))
        elif '-' in product_upper:
            code_part = product_upper.split('-')[-1]
            if code_part in answer_upper and len(code_part) >= 3:
                filtered_images.append((img_path, product_name))
    
    return filtered_images

def get_retrieval_qa(llm, db, query):
    compressor = get_reranker()
    retriever = db.as_retriever(search_kwargs={"k": 5})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    document_chain = get_document_chain(llm)
    retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
    result = retrieval_chain.invoke({"input": query})
    
    all_images_with_names = []
    for doc in result['context']:
        images_with_names = get_relevant_images_with_names(doc.metadata, doc.page_content)
        all_images_with_names.extend(images_with_names)
    
    result['images_with_names'] = filter_images_by_answer(all_images_with_names, result['answer'])
    
    return result

def data_ingestion(uploaded_files):
    clear_old_images()
    
    docs = []
    for uploaded_file in uploaded_files:
        if not os.path.exists("documents"):
            os.makedirs("documents")
        with open(os.path.join("documents", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    extract_images_from_pdfs()
    
    loader = PyPDFDirectoryLoader("documents")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs.extend(text_splitter.split_documents(data))
    return docs

def embeddings_exist():
    """Check if FAISS index already exists"""
    return os.path.exists("faiss_index1") and os.path.exists("faiss_index1/index.faiss")

def vector_embeddings(docs, force_refresh=False):
    """Create embeddings only if they don't exist or force refresh is requested"""
    embeddings = get_embeddings()
    
    if not force_refresh and embeddings_exist():
        with st.sidebar:
            st.info("✅ Using existing product catalog (embeddings found)")
        return FAISS.load_local("faiss_index1", embeddings, allow_dangerous_deserialization=True)
    
    with st.sidebar:
        st.write("📚 Creating new product embeddings...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index1")
    with st.sidebar:
        st.success("✅ Product catalog ready!")
    return db