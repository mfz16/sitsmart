"""
SitSmart Assistant - Main UI Application
Refactored version with separated concerns
"""
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Import custom modules
from utils.auth import authenticate_user
from utils.rate_limiting import initialize_rate_limiting, check_rate_limit
from utils.document_processing import (
    get_embeddings, groq_model, get_retrieval_qa, 
    embeddings_exist, vector_embeddings
)
from utils.ui_components import load_css, display_header, display_images, display_reference_materials
from admin_panel import render_admin_panel, render_customer_sidebar

# Must be first Streamlit command
st.set_page_config(
    page_title="SitSmart Assistant", 
    layout="wide",
    page_icon="🪑",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Initialize rate limiting
initialize_rate_limiting()

# Authentication check
if not authenticate_user():
    st.stop()

# Display header based on user role
display_header(st.session_state.user_role)

# Render appropriate panel based on user role
if st.session_state.user_role == "admin":
    render_admin_panel()
else:
    render_customer_sidebar()

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

    # Load model
    llm = groq_model()
    
    # Load existing product database (no auto-creation)
    db = None
    embeddings = get_embeddings()
    
    if embeddings_exist():
        try:
            db = FAISS.load_local("faiss_index1", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error accessing product database: {e}")
    
    if not db:
        st.error("❌ Our product information is currently being updated. Please contact our support team or try again later.")
        st.stop()
    
    # Show assistant thinking with placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Searching our product collection...")
        
        try:
            # Search product database
            with st.status("🔍 Searching product database...", expanded=False) as status:
                status.write("📖 Analyzing your question...")
                result = get_retrieval_qa(llm, db, prompt)
                status.write("🎯 Finding relevant products...")
                answer = result['answer']
                status.write("✅ Found information!")
                status.update(label="✅ Search complete!", state="complete")
            
            # Update with actual answer
            message_placeholder.markdown(answer)
            
            # Display images based on context
            display_images(prompt, answer, result, st.session_state.user_role)
            
            # Add to session history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Show reference materials
            display_reference_materials(result, prompt, st.session_state.user_role)
        
        except Exception as e:
            message_placeholder.markdown("❌ I'm having trouble accessing our product information right now. Please try again in a moment.")
            if st.session_state.user_role == "admin":
                st.error(f"Technical Details: {str(e)}")