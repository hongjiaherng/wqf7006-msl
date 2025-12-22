from PIL import Image
from directory import static_dir
import streamlit as st
import os

# banner image
logo = Image.open(os.path.join(static_dir, 'um_logo.png'))
st.image(logo)

# title
st.title('ABOUT US.')
st.text("We are a team of students from the University of Malaya passionate about using AI for social good. This demo showcases our Malay Sign Language AI, designed to help bridge communication between people who cannot speak and the wider world. We hope this project can empower more individuals and make everyday communication more inclusive.")

# members' profile
st.subheader('Member Profile')
col1, col2 = st.columns(2)

with col1:
    st.json({
        "name": "Hong Jia Herng",
        "major": "Artificial Intelligence"
    })
    st.json({
        "name": "Cheong Yi Fong",
        "major": "Artificial Intelligence"
    })
with col2:
    st.json({
        "name": "Khor Yin Looon",
        "major": "Artificial Intelligence"
    })
    st.json({
        "name": "Chee Zen Yu",
        "major": "Artificial Intelligence"
    })