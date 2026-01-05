import mediapipe as mp
import numpy as np
import streamlit as st
import torch, cv2, io, av
from streamlit_webrtc import VideoProcessorBase

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion from BGR to RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is no longer writeable
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)   # Color conversion RGB to BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)    # Draw right connections


def draw_styled_landmarks(image,results):
    # Draw pose connection
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1,circle_radius=1)
                              )
    # Draw left hand connection
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=1,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1,circle_radius=1)
                              )
    # Draw right hand connection
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=1,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1,circle_radius=1)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])


# device configuration
@st.cache_resource
def get_device():
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("device:", device)
    return device


# generate a subtitle file
def create_vtt(predictions, vtt_path):
    def format_timestamp(seconds):
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hrs:01}:{mins:02}:{secs:02}.{millis:03}"

    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for start, end, label in predictions:
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{label}\n\n")



@st.cache_data(show_spinner="Generating subtitles...")
def cached_subtitle_generation(_video_obj, _device, _model): 
    return generate_subtitle_from_video(_video_obj, device=_device, model=_model)


@st.cache_data(show_spinner="Processing video landmarks...")
def cached_video_processing(_video_obj):
    return generate_video_with_landmark(_video_obj)


# generate subtitle from video
def generate_subtitle_from_video(_cap, device, model):
    sequence = []
    
    # open video with OpenCV
    cap = cv2.VideoCapture(_cap)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Total frames: {frame_count}")

    sequence, predictions = [], []
    frame_count, start_frame = 0, 0
    current_gesture = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        _, results = mediapipe_detection(frame, holistic)
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
                predicted_label = model.gestures[max_idx.item()]
                confidence = float(max_val.item())

                # print(f"frames: {len(sequence)}, Prediction: {predicted_label} ({confidence:.2f})")

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
    # if current_gesture is not None:
    #     predictions.append((start_frame/fps, frame_count/fps, current_gesture))

    return predictions


# generate video with landmark
def generate_video_with_landmark(_cap):
    # open video with OpenCV
    cap = cv2.VideoCapture(_cap)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2) * 2
    height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2) * 2

    output_memory_file = io.BytesIO()
    output_container = av.open(output_memory_file, 'w', format="mp4")
    stream = output_container.add_stream('h264', rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)

        # write the frame with landmarks to the output video
        draw_styled_landmarks(image, results)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # create frame
        av_frame = av.VideoFrame.from_ndarray(image, format='rgb24')

        # encode and mux
        for packet in stream.encode(av_frame):
            output_container.mux(packet)

    # flush the encoder
    for packet in stream.encode(None):
        output_container.mux(packet)
    output_container.close()
    cap.release()

    # reset pointer to start of the file
    output_memory_file.seek(0)
    return output_memory_file


class VideoProcessor:
    def __init__(self, model, device, threshold):
        self.model = model
        self.device = device
        self.sequence = []
        self.detected_label = ""
        self.frame_count = 0
        self.threshold = threshold

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        try:
            image, results = mediapipe_detection(image, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]
            self.frame_count += 1

            if results.left_hand_landmarks or results.right_hand_landmarks:
                if len(self.sequence) == 30 and self.frame_count % 6 == 0:
                    input_data = torch.tensor(np.expand_dims(self.sequence, axis=0), dtype=torch.float32).to(self.device)
                    
                    with torch.no_grad():
                        res = self.model(input_data)
                        probabilities = torch.softmax(res, dim=1)
                        max_val, max_idx = torch.max(probabilities, dim=1)
                        predicted_label = self.model.gestures[max_idx.item()]
                        confidence = float(max_val.item())

                        self.detected_label = str(predicted_label if confidence > self.threshold else "")
                        print("threshold:", self.threshold, " Prediction:", self.detected_label, f"({confidence:.2f})")

                font = cv2.FONT_HERSHEY_SIMPLEX
                height, width, _ = image.shape
                (text_width, text_height), baseline = cv2.getTextSize(self.detected_label, font, 1, 2)
                text_x = (width // 2) - (text_width // 2)
                text_y = height - 50
                cv2.putText(image, f'{self.detected_label}', (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                self.detected_label = ""
            
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        
        except Exception as e:
            print(f"Error in callback: {e}")
            return av.VideoFrame.from_ndarray(image, format="bgr24")