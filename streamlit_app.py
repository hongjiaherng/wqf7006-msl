from directory import pages_dir
import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Fix ERROR 15

st.set_page_config(page_title="IsyaratAI")
pages = st.navigation([
    st.Page(os.path.join(pages_dir, 'index.py'), title='Home'),
    st.Page(os.path.join(pages_dir, 'demo.py'), title='Demo'),
])

pages.run()