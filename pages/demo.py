import os, torch, cv2, tempfile, io, av, warnings
import streamlit as st
import numpy as np
from directory import weight_dir
from model import load_model
from utils import (
    mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints, create_vtt
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IsyaratAI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title('DEMO.')

gestures = ['hi', 'beli', 'pukul', 'nasi_lemak',
            'lemak', 'kereta', 'nasi', 'marah',
            'anak_lelaki', 'baik', 'jangan', 'apa_khabar',
            'main', 'pinjam', 'buat', 'ribut',
            'pandai_2', 'emak_saudara', 'jahat', 'panas',
            'assalamualaikum', 'lelaki', 'bomba', 'emak',
            'sejuk', 'masalah', 'beli_2', 'anak_perempuan',
            'perempuan', 'panas_2']

# device configuration
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("device:", device)

weight_file = os.path.join(weight_dir, 'trained_model.pth')
model = load_model(device, weight_file)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# user interface setup
tab1, tab2 = st.tabs(["Upload", "Camera"])

with tab1:
    videos = st.file_uploader("Feed me with your videos", accept_multiple_files=True, type=["mp4", "mov"])

    if videos:
        st.subheader("Video Player")
        left_col, right_col = st.columns([1, 3])
        with left_col:
                video_names = [v.name for v in videos]
                selected_name = st.radio("Select a video", video_names)

        with right_col:
            if selected_name:
                sequence = []
                selected_video = next(v for v in videos if v.name == selected_name)

                # temporary file for OpenCV to read the upload
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(selected_video.read())
                
                # open video with OpenCV
                cap = cv2.VideoCapture(tfile.name)
                tfile.close()

                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Total frames: {frame_count}")

                output_memory_file = io.BytesIO()
                output_container = av.open(output_memory_file, 'w', format="mp4")
                stream = output_container.add_stream('h264', rate=int(fps))
                stream.width = width
                stream.height = height
                stream.pix_fmt = 'yuv420p'

                sequence, predictions = [], []
                frame_count, start_frame = 0, 0
                current_gesture = None

                while cap.isOpened():
                    ret, frame = cap.read()

                    if not ret:
                        break

                    image, results = mediapipe_detection(frame, holistic)

                    # write the frame with landmarks to the output video
                    draw_styled_landmarks(image, results)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    av_frame = av.VideoFrame.from_ndarray(image, format='rgb24')

                    # encode and mux
                    for packet in stream.encode(av_frame):
                        output_container.mux(packet)

                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]
                    frame_count += 1
                    
                    # perform prediction every 30 frames
                    if len(sequence) == 30:
                        input_data = torch.tensor(np.expand_dims(sequence, axis=0), dtype=torch.float32).to(device)
                        
                        with torch.no_grad():
                            res = model(input_data)
                            probabilities = torch.softmax(res, dim=1)
                            max_val, max_idx = torch.max(probabilities, dim=1)
                            predicted_label = gestures[max_idx.item()]
                            confidence = float(max_val.item())

                            print(f"frames: {len(sequence)}, Prediction: {predicted_label} ({confidence:.2f})")

                            # threshold logic
                            detected_label = predicted_label if confidence > 0.9 else None

                            if detected_label == current_gesture:
                                pass
                            else:
                                if current_gesture is not None:
                                    end_time = (frame_count - 1) / fps
                                    start_time = start_frame / fps
                                    predictions.append((start_time, end_time, current_gesture))
                                
                                current_gesture = detected_label
                                start_frame = frame_count

                # handle the last gesture after loop ends
                if current_gesture is not None:
                    predictions.append((start_frame/fps, frame_count/fps, current_gesture))

                # flush the encoder
                for packet in stream.encode(None):
                    output_container.mux(packet)
                output_container.close()
                cap.release()

                # reset pointer to start of the file
                output_memory_file.seek(0)

                # create the subtitle file
                vtt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".vtt")
                create_vtt(predictions, vtt_file.name)

                # load video and subtitle into media player
                st.video(output_memory_file, subtitles=vtt_file.name)
            else:
                st.info("No video selected")
    
with tab2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
