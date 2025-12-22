from directory import pages_dir
import streamlit as st
import os

st.set_page_config(page_title="IsyaratAI")
pages = st.navigation([
    st.Page(os.path.join(pages_dir, 'index.py'), title='Home'),
    st.Page(os.path.join(pages_dir, 'about_us.py'), title='About us'),
    st.Page(os.path.join(pages_dir, 'demo.py'), title='Demo'),
])

pages.run()