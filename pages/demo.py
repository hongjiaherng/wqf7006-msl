import os, warnings, tempfile, cv2, av
import streamlit as st
from directory import weight_dir
from model import load_model
from utils import (
    get_device, create_vtt, generate_subtitle_from_video, generate_video_with_landmark, VideoProcessor
)
from streamlit_webrtc import webrtc_streamer, WebRtcMode
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
    st.subheader("Real-time Sign Language Detection")
    st.info("Ensure your webcam is not being used by another application.")
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

    # initialize the streamer
    processor = VideoProcessor(model=model, device=device, threshold=threshold)
    webrtc_streamer(
        key="sign-language-detector",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=processor.recv,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )