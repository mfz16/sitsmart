"""UI components for SitSmart application"""
import streamlit as st
import os
from PIL import Image
from .image_processing import get_images_for_models, extract_model_names_from_text, is_image_request

def load_css():
    """Load custom CSS for beautiful UI"""
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
    
    .stImage button {
        display: none !important;
    }
    
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

def display_header(user_role):
    """Display header based on user role"""
    if user_role == "customer":
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

def display_images(prompt, answer, result, user_role):
    """Display images based on user request and context"""
    user_wants_images = is_image_request(prompt)
    
    user_models = extract_model_names_from_text(prompt)
    response_models = extract_model_names_from_text(answer)
    all_mentioned_models = list(set(user_models + response_models))
    
    # Debug info for admin
    if user_role == "admin":
        st.write(f"Debug - User wants images: {user_wants_images}")
        st.write(f"Debug - User models: {user_models}")
        st.write(f"Debug - Response models: {response_models}")
        st.write(f"Debug - All models: {all_mentioned_models}")
    
    images_to_show = []
    
    if user_wants_images or all_mentioned_models:
        if all_mentioned_models:
            images_to_show = get_images_for_models(all_mentioned_models)
            if user_role == "admin":
                st.write(f"Debug - Found {len(images_to_show)} images for models")
        
        if not images_to_show and result.get('images_with_names'):
            images_to_show = result['images_with_names'][:3]
            if user_role == "admin":
                st.write(f"Debug - Using context images: {len(images_to_show)}")
        
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
                            st.error(f"Image error: {e}")
    elif user_wants_images:
        st.info("🖼️ No images found. Please check if images have been extracted in the admin panel.")

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