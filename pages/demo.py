import os, warnings, tempfile, cv2
import streamlit as st
import numpy as np
from directory import weight_dir
from model import load_model
from utils import (
    get_device, create_vtt, cached_video_processing, cached_subtitle_generation, 
    generate_subtitle_from_video, generate_video_with_landmark
)
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IsyaratAI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title('DEMO.')

if "gestures" not in st.session_state:
    st.session_state.gestures = [
        'anak_lelaki', 'anak_perempuan', 'apa_khabar', 'assalamualaikum', 'baik',
        'beli', 'beli_2', 'bomba', 'buat', 'emak', 'emak_saudara', 'hi', 'jahat',
        'jangan', 'kereta', 'lelaki', 'lemak', 'main', 'marah', 'masalah', 'nasi',
        'nasi_lemak', 'panas', 'panas_2', 'pandai_2', 'perempuan', 'pinjam', 'pukul',
        'ribut', 'sejuk'
    ]

weight_file = os.path.join(weight_dir, 'trained_model.pth')
device = get_device()
model = load_model(device, weight_file)

# user interface setup
tab1, tab2 = st.tabs(["Upload", "Camera"])

with tab1:
    video = st.file_uploader("Feed me with your video", accept_multiple_files=False, type=["mp4", "mov"])

    if video:
        st.subheader("Video Player")
        
        with st.spinner("Processing video and generating subtitles... please wait."):
            # temporary file for OpenCV to read the upload
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video.read())

            # generate video with landmark and subtitle
            MAX_WORKERS = max(os.cpu_count() - 2, 2)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_video = executor.submit(generate_video_with_landmark, tfile.name)
                future_subtitle = executor.submit(generate_subtitle_from_video, tfile.name, device, model)
                tfile.close()

                # wait for both to finish and get the results
                output_memory_file = future_video.result()
                predictions = future_subtitle.result()

                # create the subtitle file
                vtt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".vtt")
                create_vtt(predictions, vtt_file.name)

            # load into media player
            if output_memory_file:
                st.video(output_memory_file, subtitles=vtt_file.name)
    
with tab2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
