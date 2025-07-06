import streamlit as st

# Must be first Streamlit command
st.set_page_config(
    page_title="SitSmart Assistant", 
    layout="wide",
    page_icon="🪑",
    initial_sidebar_state="expanded"
)

import os
import time
from groq import Groq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF for image extraction
from PIL import Image
import io
import base64
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
import numpy as np
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

# Initialize embeddings with caching
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

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

# Optimized prompt template (shorter for faster processing)
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

# Load Table Transformer model
@st.cache_resource
def load_table_transformer():
    """Load Table Transformer model for table detection"""
    try:
        # Check if timm is available
        import timm
        processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        return processor, model
    except ImportError:
        st.info("📋 Table Transformer not available. Using basic extraction. Install 'timm' for advanced table detection.")
        return None, None
    except Exception as e:
        st.warning(f"Could not load Table Transformer: {e}")
        return None, None

# Detect tables using Table Transformer
def detect_tables_in_page(page_image, processor, model):
    """Detect table regions in page image"""
    if processor is None or model is None:
        return []
    
    try:
        # Prepare image for model
        inputs = processor(page_image, return_tensors="pt")
        
        # Run detection
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process results
        target_sizes = torch.tensor([page_image.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
        
        # Extract table bounding boxes
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.7:  # High confidence tables only
                tables.append(box.tolist())
        
        return tables
    except Exception as e:
        print(f"Table detection error: {e}")
        return []

# Extract images from PDFs with optional Table Transformer
def extract_images_from_pdfs():
    """Extract images from PDFs with optional Table Transformer for better mapping"""
    images_data = {}
    if not os.path.exists("images"):
        os.makedirs("images")
    
    # Try to load Table Transformer
    processor, model = load_table_transformer()
    use_table_transformer = processor is not None and model is not None
    
    for filename in os.listdir("documents"):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join("documents", filename)
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get text and extract product codes
                if use_table_transformer:
                    # Advanced: Use Table Transformer
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    table_boxes = detect_tables_in_page(page_image, processor, model)
                    
                    # Get detailed text positions
                    text_dict = page.get_text("dict")
                    product_positions = []
                    
                    for block in text_dict["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    text = span["text"].strip()
                                    import re
                                    if re.match(r'SSC\s*-\s*[A-Z0-9]{2,4}', text, re.IGNORECASE):
                                        bbox = span["bbox"]
                                        product_positions.append({
                                            'name': text.upper().replace(' ', ''),
                                            'bbox': bbox
                                        })
                else:
                    # Basic: Simple text extraction
                    page_text = page.get_text()
                    import re
                    product_codes = re.findall(r'SSC\s*-\s*[A-Z0-9]{2,4}', page_text, re.IGNORECASE)
                    product_positions = [{'name': code.upper().replace(' ', ''), 'bbox': None} for code in product_codes]
                    table_boxes = []
                
                # Get images from page
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        
                        # Find best matching product
                        best_product = "Product"
                        
                        if use_table_transformer and table_boxes and product_positions:
                            # Advanced matching with table structure
                            if img_index < len(product_positions):
                                best_product = product_positions[img_index]['name']
                        elif product_positions:
                            # Basic matching
                            best_product = product_positions[img_index % len(product_positions)]['name']
                        
                        # Save image with product name
                        img_name = f"{filename}_page{page_num+1}_{best_product.replace('-', '_')}_img{img_index+1}.png"
                        img_path = os.path.join("images", img_name)
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        # Store mapping
                        key = f"{filename}_page_{page_num+1}_{best_product}"
                        if key not in images_data:
                            images_data[key] = []
                        images_data[key].append((img_path, best_product))
                    
                    pix = None
            doc.close()
    
    return images_data

# Clear old images
def clear_old_images():
    """Delete all existing images to force re-extraction"""
    if os.path.exists("images"):
        import shutil
        shutil.rmtree("images")
    os.makedirs("images", exist_ok=True)

# Document ingestion with fresh image extraction
def data_ingestion(uploaded_files):
    # Clear old images first
    clear_old_images()
    
    docs = []
    for uploaded_file in uploaded_files:
        if not os.path.exists("documents"):
            os.makedirs("documents")
        with open(os.path.join("documents", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Extract images with new logic
    extract_images_from_pdfs()
    
    loader = PyPDFDirectoryLoader("documents")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs.extend(text_splitter.split_documents(data))
    return docs

# Check if embeddings exist
def embeddings_exist():
    """Check if FAISS index already exists"""
    return os.path.exists("faiss_index1") and os.path.exists("faiss_index1/index.faiss")

# Vector DB creation (only when needed)
def vector_embeddings(docs, force_refresh=False):
    """Create embeddings only if they don't exist or force refresh is requested"""
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

# Load LLM with caching
@st.cache_resource
def groq_model():
    return ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",  # Faster model
        api_key=os.environ.get("GROQ_API_KEY")
    )

# Cache reranker model
@st.cache_resource
def get_reranker():
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    return CrossEncoderReranker(model=model, top_n=2)  # Reduced from 3 to 2

# Cache document chain
@st.cache_resource
def get_document_chain(_llm):
    return create_stuff_documents_chain(_llm, PROMPT)

# Extract product name from document content (table format)
def extract_product_name(doc_content):
    """Extract Model No. from table: empty | Model | Model No. | Description | Price"""
    lines = doc_content.split('\n')
    
    import re
    
    # Look for table structure and extract Model No. column
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Check if this line contains table headers
        if 'model no' in line.lower() or 'model' in line.lower():
            # Look at the next few lines for actual data
            for j in range(i+1, min(i+5, len(lines))):
                data_line = lines[j].strip()
                if not data_line:
                    continue
                
                # Split by common table separators
                parts = re.split(r'\s{2,}|\t|\|', data_line)
                
                # Table structure: empty | Model | Model No. | Description | Price
                # So Model No. should be in index 2 (3rd column)
                if len(parts) >= 3:
                    model_no = parts[2].strip()
                    # Check if this looks like a model number
                    if re.match(r'^[A-Z]{2,4}\s*-\s*[A-Z0-9]{2,4}$', model_no, re.IGNORECASE):
                        return model_no.upper()
    
    # Fallback: Look for SSC patterns anywhere in the content
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for SSC model patterns
        ssc_patterns = [
            r'SSC\s*-\s*[A-Z0-9]{2,4}',  # SSC-NMO, SSC -SP01
            r'SSC\s+[A-Z0-9]{2,4}',      # SSC NMO
        ]
        
        for pattern in ssc_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group().upper().replace(' ', ' ')  # Normalize spacing
    
    # Look for any model-like patterns
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Find standalone model codes
        words = line.split()
        for word in words:
            if re.match(r'^[A-Z]{2,4}-[A-Z0-9]{2,4}$', word, re.IGNORECASE):
                return word.upper()
    
    return "Product"

# Get relevant images for a document with product names
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
    
    # First try: Look for images specifically tagged with this product
    product_key = product_name.replace(' ', '_').replace('-', '_')
    
    for img_file in os.listdir("images"):
        if (img_file.startswith(filename) and 
            f"page{page}" in img_file and 
            product_key in img_file):
            img_path = os.path.join("images", img_file)
            images_with_names.append((img_path, product_name))
    
    # Fallback: If no product-specific images, use any images from the same page
    if not images_with_names:
        for img_file in os.listdir("images"):
            if img_file.startswith(filename) and f"page{page}" in img_file:
                img_path = os.path.join("images", img_file)
                images_with_names.append((img_path, product_name))
    
    return images_with_names[:1]  # Limit to 1 image per product

# Extract model names from text
def extract_model_names_from_text(text):
    """Extract SSC model names from user query or AI response"""
    import re
    if not text:
        return []
    
    # Find all SSC patterns
    patterns = [
        r'SSC\s*-\s*[A-Z0-9]{2,4}',  # SSC-NMO, SSC -SP01
        r'SSC\s+[A-Z0-9]{2,4}',      # SSC NMO
    ]
    
    found_models = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Normalize the format
            normalized = match.upper().replace(' ', '').replace('SSC-', 'SSC-').replace('SSC', 'SSC-')
            if not normalized.startswith('SSC-'):
                normalized = 'SSC-' + normalized.replace('SSC', '')
            found_models.add(normalized)
    
    return list(found_models)

# Check if user is asking to see images
def is_image_request(user_query):
    """Check if user is asking to see/show images"""
    image_keywords = ['show', 'image', 'picture', 'photo', 'see', 'look', 'display', 'view']
    query_lower = user_query.lower()
    return any(keyword in query_lower for keyword in image_keywords)

# Get images for specific model names
def get_images_for_models(model_names):
    """Get images for specific model names from the images folder"""
    if not os.path.exists("images") or not model_names:
        return []
    
    found_images = []
    all_image_files = os.listdir("images")
    
    for model_name in model_names:
        # Try multiple format variations
        model_variations = [
            model_name.replace('-', '_').replace(' ', '_'),  # SSC_NMO
            model_name.replace('-', '').replace(' ', ''),    # SSCNMO
            model_name.upper(),                              # SSC-NMO
            model_name.replace('SSC-', '').replace('SSC_', ''), # Just the code part
        ]
        
        for variation in model_variations:
            for img_file in all_image_files:
                if variation.upper() in img_file.upper():
                    img_path = os.path.join("images", img_file)
                    found_images.append((img_path, model_name))
                    break  # Found one for this variation, move to next
            if found_images:  # Found images for this model
                break
    
    return found_images[:6]  # Limit to 6 images max

# Filter images based on products mentioned in answer
def filter_images_by_answer(images_with_names, answer_text):
    """Only return images for products mentioned in the LLM answer"""
    if not images_with_names or not answer_text:
        return []
    
    filtered_images = []
    answer_upper = answer_text.upper()
    
    for img_path, product_name in images_with_names:
        product_upper = product_name.upper()
        
        # Check if product code is mentioned in answer
        if product_upper in answer_upper:
            filtered_images.append((img_path, product_name))
        # Also check for partial matches (e.g., "SSC-NMO" mentioned as "NMO")
        elif '-' in product_upper:
            code_part = product_upper.split('-')[-1]  # Get part after dash
            if code_part in answer_upper and len(code_part) >= 3:
                filtered_images.append((img_path, product_name))
    
    return filtered_images

# Optimized retrieval-based QA with filtered images
def get_retrieval_qa(llm, db, query):
    compressor = get_reranker()
    retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more initially
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    document_chain = get_document_chain(llm)
    retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
    result = retrieval_chain.invoke({"input": query})
    
    # Collect all available images with names
    all_images_with_names = []
    for doc in result['context']:
        images_with_names = get_relevant_images_with_names(doc.metadata, doc.page_content)
        all_images_with_names.extend(images_with_names)
    
    # Filter images based on what's mentioned in the answer
    result['images_with_names'] = filter_images_by_answer(all_images_with_names, result['answer'])
    
    return result


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
    
    /* Product image styling */
    .stImage {
        max-width: 200px !important;
        max-height: 200px !important;
        object-fit: contain !important;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    
    .stImage > img {
        max-width: 200px !important;
        max-height: 200px !important;
        object-fit: contain !important;
        cursor: default !important;
    }
    
    /* Disable image click/zoom */
    .stImage button {
        display: none !important;
    }
    
    /* Product image container */
    .product-image-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 0.5rem;
        border-radius: 8px;
        background-color: #fafafa;
        margin: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 🧠 Streamlit UI starts here
# -------------------------------

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
        
        if st.button("🔄 Update Product Catalog", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("📚 Processing documents and extracting images..."):
                    # Clear cache to force fresh processing
                    st.cache_resource.clear()
                    docs = data_ingestion(uploaded_files)
                    vector_embeddings(docs, force_refresh=True)  # Force new embeddings for new docs
                st.success("✅ Product catalog and images updated successfully!")
        
        # Add button to force image re-extraction from existing documents
        if st.button("🖼️ Re-extract Images", use_container_width=True):
            if os.path.exists("documents") and os.listdir("documents"):
                with st.spinner("🖼️ Re-extracting images with new logic..."):
                    clear_old_images()
                    st.cache_resource.clear()
                    extract_images_from_pdfs()
                st.success("✅ Images re-extracted successfully!")
            else:
                st.warning("No documents found. Please upload PDFs first.")
        
        # Add button to refresh embeddings only
        if st.button("🔄 Refresh Embeddings Only", use_container_width=True):
            if os.path.exists("documents") and os.listdir("documents"):
                with st.spinner("🔄 Creating fresh embeddings..."):
                    loader = PyPDFDirectoryLoader("documents")
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = text_splitter.split_documents(data)
                    vector_embeddings(docs, force_refresh=True)
                st.success("✅ Embeddings refreshed successfully!")
            else:
                st.warning("No documents found. Please upload PDFs first.")
        
        # Product Gallery for testing mappings
        st.subheader("🖼️ Product Gallery")
        
        if os.path.exists("images") and os.listdir("images"):
            # Get all unique product names from image filenames
            product_names = set()
            for img_file in os.listdir("images"):
                # Extract product name from filename: filename_pageX_PRODUCT_NAME_imgY.png
                parts = img_file.split('_')
                if len(parts) >= 4:
                    # Find product part (between page and img)
                    for i, part in enumerate(parts):
                        if part.startswith('page') and i + 1 < len(parts):
                            # Get everything between pageX and imgY
                            product_parts = []
                            for j in range(i + 1, len(parts)):
                                if parts[j].startswith('img'):
                                    break
                                product_parts.append(parts[j])
                            if product_parts:
                                product_name = '_'.join(product_parts).replace('_', '-')
                                if product_name != 'Product':
                                    product_names.add(product_name)
            
            if product_names:
                selected_product = st.selectbox(
                    "Select Product to View:", 
                    sorted(list(product_names)),
                    key="product_gallery"
                )
                
                if selected_product:
                    # Find images for selected product
                    product_key = selected_product.replace('-', '_')
                    product_images = []
                    
                    for img_file in os.listdir("images"):
                        if product_key in img_file:
                            img_path = os.path.join("images", img_file)
                            product_images.append(img_path)
                    
                    if product_images:
                        st.markdown(f"**Images for {selected_product}:**")
                        cols = st.columns(min(len(product_images), 3))
                        
                        for idx, img_path in enumerate(product_images):
                            with cols[idx % 3]:
                                try:
                                    img = Image.open(img_path)
                                    img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                                    st.image(img, caption=f"{selected_product}", width=200)
                                    st.caption(f"File: {os.path.basename(img_path)}")
                                except Exception as e:
                                    st.error(f"Error loading image: {e}")
                    else:
                        st.info(f"No images found for {selected_product}")
            else:
                st.info("No product images found. Please re-extract images first.")
        else:
            st.info("No images folder found. Please upload and process documents first.")
        
        # Activity stats
        st.subheader("📊 Activity Overview")
        if "messages" in st.session_state:
            st.metric("Total Conversations", len(st.session_state.messages))
        
        # Usage info (simplified)
        remaining = max(0, RATE_LIMIT_REQUESTS - len(st.session_state.rate_limit_requests))
        st.metric("Questions Left", remaining)
        
        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Manual Image Correction
        st.subheader("✏️ Manual Image Correction")
        
        if os.path.exists("images") and os.listdir("images"):
            # Get all images
            all_images = [f for f in os.listdir("images") if f.endswith('.png')]
            
            if all_images:
                # Select image to correct
                selected_image = st.selectbox(
                    "Select Image to Correct:",
                    sorted(all_images),
                    key="image_correction"
                )
                
                if selected_image:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Show current image
                        img_path = os.path.join("images", selected_image)
                        try:
                            img = Image.open(img_path)
                            img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                            st.image(img, caption="Current Image", width=150)
                        except Exception:
                            st.error("Cannot load image")
                    
                    with col2:
                        # Extract current product from filename
                        current_product = "Unknown"
                        parts = selected_image.split('_')
                        for i, part in enumerate(parts):
                            if part.startswith('page') and i + 1 < len(parts):
                                product_parts = []
                                for j in range(i + 1, len(parts)):
                                    if parts[j].startswith('img'):
                                        break
                                    product_parts.append(parts[j])
                                if product_parts:
                                    current_product = '_'.join(product_parts).replace('_', '-')
                                break
                        
                        st.write(f"**Current Product:** {current_product}")
                        st.write(f"**Filename:** {selected_image}")
                        
                        # Input new product name
                        new_product = st.text_input(
                            "Correct Product Name (e.g., SSC-NMO):",
                            value=current_product,
                            key="new_product_name"
                        )
                        
                        if st.button("Update Image Mapping", key="update_mapping"):
                            if new_product and new_product != current_product:
                                # Create new filename
                                old_path = os.path.join("images", selected_image)
                                
                                # Replace product part in filename
                                new_filename = selected_image
                                for i, part in enumerate(parts):
                                    if part.startswith('page') and i + 1 < len(parts):
                                        # Rebuild filename with new product
                                        new_parts = parts[:i+1]  # Keep up to page
                                        new_parts.append(new_product.replace('-', '_'))  # Add new product
                                        # Find and add img part
                                        for j in range(i + 1, len(parts)):
                                            if parts[j].startswith('img'):
                                                new_parts.extend(parts[j:])
                                                break
                                        new_filename = '_'.join(new_parts)
                                        break
                                
                                new_path = os.path.join("images", new_filename)
                                
                                try:
                                    os.rename(old_path, new_path)
                                    st.success(f"✅ Updated: {selected_image} → {new_filename}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating: {e}")
                            else:
                                st.warning("Please enter a different product name")
            else:
                st.info("No images found")
        else:
            st.info("No images folder found. Please extract images first.")
        
        # Image mapping diagnostics
        with st.expander("🔍 Image Mapping Diagnostics"):
            if os.path.exists("images"):
                all_images = os.listdir("images")
                st.write(f"**Total Images:** {len(all_images)}")
                
                if all_images:
                    st.write("**Image Files:**")
                    for img in sorted(all_images)[:10]:  # Show first 10
                        st.text(img)
                    if len(all_images) > 10:
                        st.text(f"... and {len(all_images) - 10} more")
            else:
                st.write("No images folder found.")
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
if prompt := st.chat_input("Ask me about SitSmart chairs, pricing, or recommendations... (Try: 'Show me SSC-NMO')"):
    # Check rate limit
    if not check_rate_limit():
        st.markdown("""
        <div class="rate-limit-warning">
            ⚠️ <strong>Please slow down!</strong> You're asking questions too quickly. 
            Please wait a moment before sending another message.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Show user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Load model and index (cached)
    llm = groq_model()
    
    # Load existing embeddings if available
    db = None
    if embeddings_exist():
        try:
            db = FAISS.load_local("faiss_index1", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
    else:
        # Create embeddings from existing documents if available
        if os.path.exists("documents") and os.listdir("documents"):
            with st.spinner("📚 Creating initial embeddings..."):
                loader = PyPDFDirectoryLoader("documents")
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(data)
                db = vector_embeddings(docs, force_refresh=False)
    
    if not db:
        st.error("❌ Product catalog is not available. Please contact our support team.")
        st.stop()
    
    # Show assistant thinking with placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Let me check our catalog...")
        
        try:
            # Run retrieval QA
            result = get_retrieval_qa(llm, db, prompt)
            answer = result['answer']
            
            # Update with actual answer
            message_placeholder.markdown(answer)
            
            # Check if user is asking to see images
            user_wants_images = is_image_request(prompt)
            
            # Extract model names from user query and AI response
            user_models = extract_model_names_from_text(prompt)
            response_models = extract_model_names_from_text(answer)
            all_mentioned_models = list(set(user_models + response_models))
            
            # Debug info for admin
            if st.session_state.user_role == "admin":
                st.write(f"Debug - User wants images: {user_wants_images}")
                st.write(f"Debug - User models: {user_models}")
                st.write(f"Debug - Response models: {response_models}")
                st.write(f"Debug - All models: {all_mentioned_models}")
            
            # Display images based on context
            images_to_show = []
            
            if user_wants_images or all_mentioned_models:
                # User asked for images OR models were mentioned
                if all_mentioned_models:
                    images_to_show = get_images_for_models(all_mentioned_models)
                    if st.session_state.user_role == "admin":
                        st.write(f"Debug - Found {len(images_to_show)} images for models")
                
                # If no specific model images, try context-based
                if not images_to_show and result.get('images_with_names'):
                    images_to_show = result['images_with_names'][:3]
                    if st.session_state.user_role == "admin":
                        st.write(f"Debug - Using context images: {len(images_to_show)}")
                
                # Show any available images if user explicitly asked
                if not images_to_show and user_wants_images and os.path.exists("images"):
                    # Show first few available images
                    all_imgs = [f for f in os.listdir("images") if f.endswith('.png')][:3]
                    for img_file in all_imgs:
                        img_path = os.path.join("images", img_file)
                        # Extract product name from filename
                        parts = img_file.split('_')
                        product_name = "Product"
                        for i, part in enumerate(parts):
                            if part.startswith('page') and i + 1 < len(parts):
                                product_parts = []
                                for j in range(i + 1, len(parts)):
                                    if parts[j].startswith('img'):
                                        break
                                    product_parts.append(parts[j])
                                if product_parts:
                                    product_name = '_'.join(product_parts).replace('_', '-')
                                break
                        images_to_show.append((img_path, product_name))
            
            elif result.get('images_with_names'):
                # Standard context-based images
                images_to_show = result['images_with_names'][:3]
            
            # Display the images
            if images_to_show:
                if user_wants_images:
                    st.markdown("**📸 Requested Product Images:**")
                else:
                    st.markdown("**📸 Product Images:**")
                
                cols = st.columns(min(len(images_to_show), 3))
                for idx, (img_path, product_name) in enumerate(images_to_show):
                    if os.path.exists(img_path):
                        with cols[idx % 3]:
                            try:
                                img = Image.open(img_path)
                                # Resize image to uniform size
                                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                                st.image(img, caption=product_name, width=200)
                            except Exception as e:
                                if st.session_state.user_role == "admin":
                                    st.error(f"Image error: {e}")
            elif user_wants_images:
                st.info("🖼️ No images found. Please check if images have been extracted in the admin panel.")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Show reference materials (only for admin or if customer specifically asks)
            if st.session_state.user_role == "admin" or "source" in prompt.lower():
                with st.expander("📚 Reference Information"):
                    for i, doc in enumerate(result['context'], 1):
                        st.markdown(f"**📄 Catalog {i}: {doc.metadata.get('source', 'Product Guide')}, Page {doc.metadata.get('page', '?')}**")
                        st.markdown(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        
                        # Show images for this document with product names
                        doc_images_with_names = get_relevant_images_with_names(doc.metadata, doc.page_content)
                        if doc_images_with_names:
                            st.markdown("**Images from this page:**")
                            img_cols = st.columns(len(doc_images_with_names))
                            for idx, (img_path, product_name) in enumerate(doc_images_with_names):
                                if os.path.exists(img_path):
                                    with img_cols[idx]:
                                        try:
                                            img = Image.open(img_path)
                                            # Resize image to uniform size
                                            img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                                            st.image(img, caption=product_name, width=200)
                                        except Exception:
                                            pass
                        
                        if i < len(result['context']):
                            st.markdown("---")
        
        except Exception as e:
            message_placeholder.markdown("❌ I'm having trouble right now. Please try asking again in a moment.")
            if st.session_state.user_role == "admin":
                st.error(f"System Error: {str(e)}")
