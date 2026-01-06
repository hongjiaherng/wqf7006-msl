import mediapipe as mp
import numpy as np
import streamlit as st
import torch, cv2, io, av, threading
from collections import deque

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
# holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as local_holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            _, results = mediapipe_detection(frame, local_holistic)
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
                    detected_label = predicted_label if confidence > 0.95 else None

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

    return predictions


# generate video with landmark
def generate_video_with_landmark(_cap):
    mp_holistic = mp.solutions.holistic

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

    with mp_holistic.Holistic(min_detection_confidence=0.5) as local_holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            image, results = mediapipe_detection(frame, local_holistic)

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
    def __init__(self, model, device, threshold, frame_skip):
        self.model = model
        self.device = device
        self.mp_holistic = mp.solutions.holistic
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5)
        self.frame_skip = frame_skip
        self.threshold = threshold

        # buffer for keypoints
        self.sequence = deque(maxlen=30 * frame_skip)
        self.frame_count = 0

        # threading for non-blocking inference
        self.detected_label = ""
        self.confidence = 0.0
        self.lock = threading.Lock()
        self.is_processing = False


    def __del__(self):
        if hasattr(self, 'holistic'):
            self.holistic.close()


    def inference(self, sequence):
        with torch.no_grad():
            input_data = torch.tensor(np.expand_dims(sequence, axis=0), dtype=torch.float32).to(self.device)
            res = self.model(input_data)
            probabilities = torch.softmax(res, dim=1)
            max_val, max_idx = torch.max(probabilities, dim=1)

            with self.lock:
                self.confidence = float(max_val.item())
                if self.confidence > self.threshold:
                    self.detected_label = self.model.gestures[max_idx.item()]
                else:
                    self.detected_label = ""
            
            print("threshold:", self.threshold, " Prediction:", self.detected_label, f"({self.confidence:.2f})")
            self.is_processing = False


    def draw_subtitle(self, image, text, confidence):
        height, width, _ = image.shape
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        thickness = 2
        
        # formatting text
        display_text = f"{text} ({confidence:.2%})" if text else "No Gesture"
        (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
        
        # draw semi-transparent background box
        padding = 20
        box_coords = ((0, height - 80), (width, height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], (0, 0, 0), -1)

        # apply transparency (alpha 0.6)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # center the text
        text_x = (width // 2) - (text_width // 2)
        text_y = height - 35
        
        # shadow for text
        cv2.putText(image, display_text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # main text (green -> confident, white -> not confident)
        color = (0, 255, 0) if confidence > self.threshold else (255, 255, 255)
        cv2.putText(image, display_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)


    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        self.frame_count += 1
        
        try:
            results = self.holistic.process(image)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            self.sequence.append(keypoints)

            if len(self.sequence) == (30 * self.frame_skip) and not self.is_processing:
                if self.frame_count % self.frame_skip == 0:
                    
                    # check if hands are actually visible before bothering the GPU
                    if results.left_hand_landmarks or results.right_hand_landmarks:
                        self.is_processing = True
                        
                        # sample the sequence based on frame_skip
                        sequence_list = list(self.sequence)[::self.frame_skip]
                        
                        # start background thread
                        thread = threading.Thread(target=self.inference, args=(sequence_list,))
                        thread.start()
                    
                with self.lock:
                    self.draw_subtitle(image, self.detected_label, self.confidence)

            return av.VideoFrame.from_ndarray(image, format="bgr24")
    
        except Exception as e:
            print(f"Error in callback: {e}")
            return av.VideoFrame.from_ndarray(image, format="bgr24")