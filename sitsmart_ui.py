import os
import time
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
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Load API key and admin password
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
admin_password = st.secrets.get("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", "admin123"))

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 10  # requests per minute
RATE_LIMIT_WINDOW = 60    # seconds

# Initialize rate limiting in session state
if "rate_limit_requests" not in st.session_state:
    st.session_state.rate_limit_requests = []

def check_rate_limit():
    """Check if user has exceeded rate limit"""
    now = datetime.now()
    # Remove requests older than the window
    st.session_state.rate_limit_requests = [
        req_time for req_time in st.session_state.rate_limit_requests 
        if now - req_time < timedelta(seconds=RATE_LIMIT_WINDOW)
    ]
    
    if len(st.session_state.rate_limit_requests) >= RATE_LIMIT_REQUESTS:
        return False
    
    st.session_state.rate_limit_requests.append(now)
    return True

# Enhanced prompt template
prompt_template = """You are SitSmart Assistant, a knowledgeable and friendly customer service representative for SitSmart India, a premium furniture company specializing in ergonomic chairs and seating solutions.

**Your Role:**
- Provide detailed, accurate information about SitSmart's chair catalog
- Help customers find the perfect chair for their needs
- Offer professional recommendations based on customer requirements
- Maintain a warm, helpful, and professional tone

**Guidelines:**
1. Always greet customers warmly and thank them for their interest in SitSmart
2. Provide specific product details including models, features, prices, and specifications when available
3. If asked about comparisons, highlight the unique benefits of each chair
4. For pricing queries, provide exact prices from the catalog when available
5. If information isn't in the context, politely say "I don't have that specific information in our current catalog, but I'd be happy to help you contact our sales team for more details."
6. Always end responses by asking if there's anything else you can help with
7. Use bullet points for features and specifications to improve readability

**Context Information:**
{context}

**Customer Question:** {input}

**Your Response:**"""

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


# Authentication function
def authenticate_user():
    """Simple authentication system"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_role = None
    
    if not st.session_state.authenticated:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: #2E86AB; margin-bottom: 2rem;">🪑 Welcome to SitSmart</h1>
            <p style="font-size: 1.2rem; color: #666;">Please select your access level</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            role = st.selectbox("Select Role:", ["Customer", "Admin"], key="role_select")
            
            if role == "Admin":
                entered_password = st.text_input("Admin Password:", type="password")
                if st.button("Login as Admin", use_container_width=True):
                    if entered_password == admin_password:
                        st.session_state.authenticated = True
                        st.session_state.user_role = "admin"
                        st.rerun()
                    else:
                        st.error("Invalid password")
            else:
                if st.button("Continue as Customer", use_container_width=True):
                    st.session_state.authenticated = True
                    st.session_state.user_role = "customer"
                    st.rerun()
        return False
    return True

# Custom CSS for beautiful UI
def load_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    
    .stChatMessage {
        background-color: #191919;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2E86AB;
    }
    
    .stChatInputContainer {
        border-top: 2px solid #e0e0e0;
        padding-top: 1rem;
    }
    
    .chat-header {
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .admin-panel {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff6b6b;
    }
    
    .rate-limit-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 1rem 0;
        color: #856404;
    }
    
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 Streamlit UI starts here
# -------------------------------

st.set_page_config(
    page_title="SitSmart Assistant", 
    layout="wide",
    page_icon="🪑",
    initial_sidebar_state="collapsed" if "user_role" in st.session_state and st.session_state.user_role == "customer" else "expanded"
)

load_css()

# Authentication check
if not authenticate_user():
    st.stop()

# Header based on user role
if st.session_state.user_role == "customer":
    st.markdown("""
    <div class="chat-header">
        <h1>🪑 SitSmart Assistant</h1>
        <p>Find your perfect chair with our AI-powered assistant</p>
        <small>Powered by advanced AI • Available 24/7</small>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="chat-header">
        <h1>🛠️ SitSmart Admin Panel</h1>
        <p>Manage documents and monitor system performance</p>
    </div>
    """, unsafe_allow_html=True)

# Admin panel in sidebar (only for admin users)
if st.session_state.user_role == "admin":
    with st.sidebar:
        st.markdown('<div class="admin-panel">', unsafe_allow_html=True)
        st.subheader("🛠️ Admin Controls")
        
        # Document management
        st.subheader("📄 Document Management")
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        
        if st.button("🔄 Update Vector DB Index", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing documents..."):
                    docs = data_ingestion(uploaded_files)
                    vector_embeddings(docs)
                st.success("✅ Documents processed successfully!")
        
        # System stats
        st.subheader("📊 System Stats")
        if "messages" in st.session_state:
            st.metric("Total Messages", len(st.session_state.messages))
        
        # Rate limit info
        remaining_requests = max(0, RATE_LIMIT_REQUESTS - len(st.session_state.rate_limit_requests))
        st.metric("API Requests Remaining", remaining_requests)
        
        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Customer view - minimal sidebar with logout only
    with st.sidebar:
        st.markdown("### 👋 Welcome, Customer!")
        st.info("💡 **Tip:** Ask me about chair specifications, pricing, or recommendations!")
        
        if st.button("🚪 Switch User", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.rerun()

# Initialize session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input with rate limiting
if prompt := st.chat_input("Ask me about SitSmart chairs, pricing, or recommendations..."):
    # Check rate limit
    if not check_rate_limit():
        st.markdown("""
        <div class="rate-limit-warning">
            ⚠️ <strong>Rate limit exceeded!</strong> Please wait a moment before sending another message.
            You can send up to 10 messages per minute.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Show user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Load model and index
    with st.spinner("🤔 Thinking..."):
        llm = groq_model()
        try:
            db = FAISS.load_local("faiss_index1", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            db = None
            st.error("❌ No product catalog found. Please contact our admin to update the system.")

        if db:
            try:
                # Run retrieval QA
                result = get_retrieval_qa(llm, db, prompt)
                answer = result['answer']

                # Show assistant's message
                with st.chat_message("assistant"):
                    st.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Show source documents (only for admin or if customer specifically asks)
                if st.session_state.user_role == "admin" or "source" in prompt.lower():
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(result['context'], 1):
                            st.markdown(f"**📄 Document {i}: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', '?')}**")
                            st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            if i < len(result['context']):
                                st.markdown("---")
            
            except Exception as e:
                st.error(f"❌ Sorry, I encountered an error while processing your request. Please try again.")
                if st.session_state.user_role == "admin":
                    st.error(f"Debug info: {str(e)}")
