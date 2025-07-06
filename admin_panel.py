"""Admin panel components for SitSmart application"""
import streamlit as st
import os
from PIL import Image
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.document_processing import data_ingestion, vector_embeddings, get_embeddings
from utils.image_processing import clear_old_images, extract_images_from_pdfs
from utils.rate_limiting import get_remaining_requests

def render_admin_panel():
    """Render the complete admin panel"""
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
                    st.cache_resource.clear()
                    docs = data_ingestion(uploaded_files)
                    vector_embeddings(docs, force_refresh=True)
                st.success("✅ Product catalog and images updated successfully!")
        
        if st.button("🖼️ Re-extract Images", use_container_width=True):
            if os.path.exists("documents") and os.listdir("documents"):
                with st.spinner("🖼️ Re-extracting images with new logic..."):
                    clear_old_images()
                    st.cache_resource.clear()
                    extract_images_from_pdfs()
                st.success("✅ Images re-extracted successfully!")
            else:
                st.warning("No documents found. Please upload PDFs first.")
        
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
        
        # Activity stats
        st.subheader("📊 Activity Overview")
        if "messages" in st.session_state:
            st.metric("Total Conversations", len(st.session_state.messages))
        
        remaining = get_remaining_requests()
        st.metric("Questions Left", remaining)
        
        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Product Gallery
    render_product_gallery()
    
    # Manual Image Correction
    render_image_correction()
    
    # Diagnostics
    render_diagnostics()

def render_product_gallery():
    """Render product gallery for testing mappings"""
    st.subheader("🖼️ Product Gallery")
    
    if os.path.exists("images") and os.listdir("images"):
        product_names = set()
        for img_file in os.listdir("images"):
            parts = img_file.split('_')
            if len(parts) >= 4:
                for i, part in enumerate(parts):
                    if part.startswith('page') and i + 1 < len(parts):
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

def render_image_correction():
    """Render manual image correction interface"""
    st.subheader("✏️ Manual Image Correction")
    
    if os.path.exists("images") and os.listdir("images"):
        all_images = [f for f in os.listdir("images") if f.endswith('.png')]
        
        if all_images:
            selected_image = st.selectbox(
                "Select Image to Correct:",
                sorted(all_images),
                key="image_correction"
            )
            
            if selected_image:
                col1, col2 = st.columns([1, 2])
                
                with col1:
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
                    
                    new_product = st.text_input(
                        "Correct Product Name (e.g., SSC-NMO):",
                        value=current_product,
                        key="new_product_name"
                    )
                    
                    if st.button("Update Image Mapping", key="update_mapping"):
                        if new_product and new_product != current_product:
                            old_path = os.path.join("images", selected_image)
                            
                            new_filename = selected_image
                            for i, part in enumerate(parts):
                                if part.startswith('page') and i + 1 < len(parts):
                                    new_parts = parts[:i+1]
                                    new_parts.append(new_product.replace('-', '_'))
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

def render_diagnostics():
    """Render diagnostics information"""
    with st.expander("🔍 Image Mapping Diagnostics"):
        if os.path.exists("images"):
            all_images = os.listdir("images")
            st.write(f"**Total Images:** {len(all_images)}")
            
            if all_images:
                st.write("**Image Files:**")
                for img in sorted(all_images)[:10]:
                    st.text(img)
                if len(all_images) > 10:
                    st.text(f"... and {len(all_images) - 10} more")
        else:
            st.write("No images folder found.")

def render_customer_sidebar():
    """Render customer sidebar"""
    with st.sidebar:
        st.markdown("### 👋 Welcome, Customer!")
        st.info("💡 **Tip:** Ask me about chair specifications, pricing, or recommendations!")
        
        if st.button("🚪 Switch User", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.rerun()