"""UI components for SitSmart application"""
import streamlit as st
import os
from PIL import Image
from .image_processing import get_images_for_models, extract_model_names_from_text, is_image_request

def load_css():
    """Load custom CSS matching SitSmart website design"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding-top: 0;
        font-family: 'Inter', sans-serif;
        color: #1f2937;
    }
    
    /* Ensure all text is readable */
    * {
        color: inherit;
    }
    
    /* Main content text */
    .main .block-container {
        color: #1f2937;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat Messages */
    .stChatMessage {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
        color: #1f2937 !important;
    }
    
    .stChatMessage:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }
    
    /* Chat Message Text */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #1f2937 !important;
    }
    
    /* User Messages */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-color: #3b82f6;
        color: #1e40af !important;
    }
    
    .stChatMessage[data-testid="user-message"] p,
    .stChatMessage[data-testid="user-message"] div,
    .stChatMessage[data-testid="user-message"] span {
        color: #1e40af !important;
    }
    
    /* Assistant Messages */
    .stChatMessage[data-testid="assistant-message"] {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #10b981;
        color: #065f46 !important;
    }
    
    .stChatMessage[data-testid="assistant-message"] p,
    .stChatMessage[data-testid="assistant-message"] div,
    .stChatMessage[data-testid="assistant-message"] span {
        color: #065f46 !important;
    }
    
    /* Chat Input */
    .stChatInputContainer {
        border-top: 1px solid #e2e8f0;
        padding-top: 1.5rem;
        background: linear-gradient(180deg, transparent 0%, #f8fafc 100%);
    }
    
    /* SitSmart Header */
    .sitsmart-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(30, 64, 175, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .sitsmart-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.1;
    }
    
    .sitsmart-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .sitsmart-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
        position: relative;
        z-index: 1;
    }
    
    /* Admin Panel */
    .admin-panel {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    
    /* Product Images */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 2px solid #e2e8f0;
    }
    
    .stImage:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        border-color: #3b82f6;
    }
    
    .stImage > img {
        max-width: 200px !important;
        max-height: 200px !important;
        object-fit: contain !important;
        cursor: default !important;
    }
    
    .stImage button {
        display: none !important;
    }
    
    /* Status Messages */
    .rate-limit-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #92400e;
        font-weight: 500;
    }
    
    /* Expanders */
    .stExpander {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        margin-top: 1rem;
        background: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Selectbox */
    .stSelectbox > div > div > select {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1rem;
        background: #f8fafc;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        color: #065f46;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        color: #991b1b;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        color: #1e40af;
    }
    
    /* Product Grid */
    .product-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 1rem 0;
    }
    
    .product-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header(user_role):
    """Display header based on user role with SitSmart branding"""
    if user_role == "customer":
        st.markdown("""
        <div class="sitsmart-header">
            <h1>🪑 SitSmart Assistant</h1>
            <p>Your AI-powered furniture consultant • Find the perfect chair for your space</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sitsmart-header">
            <h1>🛠️ SitSmart Admin Console</h1>
            <p>Manage your product catalog and customer experience</p>
        </div>
        """, unsafe_allow_html=True)

def display_images(prompt, answer, result, user_role):
    """Display images based on user request and context"""
    user_wants_images = is_image_request(prompt)
    
    user_models = extract_model_names_from_text(prompt)
    response_models = extract_model_names_from_text(answer)
    all_mentioned_models = list(set(user_models + response_models))
    
    # Technical info for admin
    if user_role == "admin":
        st.write(f"Technical - Image request detected: {user_wants_images}")
        st.write(f"Technical - Models in question: {user_models}")
        st.write(f"Technical - Models in response: {response_models}")
        st.write(f"Technical - All mentioned models: {all_mentioned_models}")
    
    images_to_show = []
    
    if user_wants_images or all_mentioned_models:
        if all_mentioned_models:
            images_to_show = get_images_for_models(all_mentioned_models)
            if user_role == "admin":
                st.write(f"Technical - Found {len(images_to_show)} images for models")
        
        if not images_to_show and result.get('images_with_names'):
            images_to_show = result['images_with_names'][:3]
            if user_role == "admin":
                st.write(f"Technical - Using context images: {len(images_to_show)}")
        
        if not images_to_show and user_wants_images and os.path.exists("images"):
            all_imgs = [f for f in os.listdir("images") if f.endswith('.png')][:3]
            for img_file in all_imgs:
                img_path = os.path.join("images", img_file)
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
        images_to_show = result['images_with_names'][:3]
    
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
                        img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                        st.image(img, caption=product_name, width=200)
                    except Exception as e:
                        if user_role == "admin":
                            st.error(f"Technical - Image loading error: {e}")
    elif user_wants_images:
        st.info("🖼️ No product images available at the moment. Our team is working to add more visual content.")

def display_reference_materials(result, prompt, user_role):
    """Display reference materials for admin or when requested"""
    from .document_processing import get_relevant_images_with_names
    
    if user_role == "admin" or "source" in prompt.lower():
        with st.expander("📚 Reference Information"):
            for i, doc in enumerate(result['context'], 1):
                st.markdown(f"**📄 Catalog {i}: {doc.metadata.get('source', 'Product Guide')}, Page {doc.metadata.get('page', '?')}**")
                st.markdown(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                
                doc_images_with_names = get_relevant_images_with_names(doc.metadata, doc.page_content)
                if doc_images_with_names:
                    st.markdown("**Images from this page:**")
                    img_cols = st.columns(len(doc_images_with_names))
                    for idx, (img_path, product_name) in enumerate(doc_images_with_names):
                        if os.path.exists(img_path):
                            with img_cols[idx]:
                                try:
                                    img = Image.open(img_path)
                                    img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                                    st.image(img, caption=product_name, width=200)
                                except Exception:
                                    pass
                
                if i < len(result['context']):
                    st.markdown("---")