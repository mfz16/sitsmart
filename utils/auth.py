"""Authentication utilities for SitSmart application"""
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

def get_admin_password():
    """Get admin password from secrets or environment"""
    return st.secrets.get("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", "admin123"))

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
                    if entered_password == get_admin_password():
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