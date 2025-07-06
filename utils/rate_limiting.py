"""Rate limiting utilities for SitSmart application"""
import streamlit as st
from datetime import datetime, timedelta

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 10  # requests per minute
RATE_LIMIT_WINDOW = 60    # seconds

def initialize_rate_limiting():
    """Initialize rate limiting in session state"""
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

def get_remaining_requests():
    """Get number of remaining requests"""
    return max(0, RATE_LIMIT_REQUESTS - len(st.session_state.rate_limit_requests))