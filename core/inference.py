# inference.py
from openvino.runtime import Core
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import time
import collections
import sys
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

from core.email_alert import send_email_alert
from notebook_utils import VideoPlayer
from core.preprocessing import display_text_fnc

sys.path.append(os.path.abspath("Project"))

# Load the model
decoder = load_model(r"C:/Users/malle/Desktop/Proj/model/classifier_lstm_e19.h5", compile=False)
decoder.summary()

# Load OpenVINO IR Model
ie = Core()
encoder_model_ir = ie.read_model(model=r"C:/Users/malle/Desktop/Proj/inceptionv3_model/saved_model.xml")
compiled_encoder = ie.compile_model(model=encoder_model_ir, device_name="CPU")
output_layer_ir = compiled_encoder.output(0)

IMG_SIZE = (299, 224)
BATCH_SIZE = 30
EPOCHS = 100
MAX_SEQ_LENGTH = 32
NUM_FEATURES = 2048

batch_size, height_en, width_en, channels = compiled_encoder.inputs[0].shape
frames2decode = list(decoder.inputs[0].shape)[1]

class_vocab = ['Abuse', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting']
frames_idx = []


def run_action_recognition(video_label, source='0', flip=True, skip_first_frames=0): #, should_stop=lambda: False'''
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)

    if not cap.isOpened():
        print("Failed to open video source.")
        return

    # Skip frames if needed
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)

    encoder_output = []
    decoded_labels = ['Normal'] * 3
    decoded_top_probs = [0.0] * 3
    counter = 0
    processing_times = collections.deque()

    def update_frame():
        nonlocal counter, encoder_output, decoded_labels, decoded_top_probs

        # if should_stop():
        #     cap.release()
        #     return
        
        ret, frame = cap.read()
        if not ret:
            print("Source ended or failed to read.")
            cap.release()
            return

        counter += 1

        if flip:
            frame = cv2.flip(frame, 1)

        preprocessed = cv2.resize(frame, IMG_SIZE)
        preprocessed = preprocessed[:, :, [2, 1, 0]]  # BGR to RGB

        if counter % 2 == 0:
            start_time = time.time()

            encoder_output.append(compiled_encoder([preprocessed[None, ...]])[output_layer_ir][0])

            if len(encoder_output) == frames2decode:
                encoder_output_np = np.array(encoder_output)[None, ...]
                probabilities = decoder.predict(encoder_output_np)[0]

                top_indices = np.argsort(probabilities)[::-1][:3]
                for i, idx in enumerate(top_indices):
                    decoded_labels[i] = class_vocab[idx]
                    decoded_top_probs[i] = probabilities[idx]

                encoder_output = []

                if decoded_labels[0] != "Normal":
                    send_email_alert(decoded_labels[0], decoded_top_probs[0] * 100)

            end_time = time.time()
            processing_times.append(end_time - start_time)

            if len(processing_times) > 200:
                processing_times.popleft()

        processing_time = np.mean(processing_times) * 1000 if processing_times else 0

        # Display predictions
        frame = cv2.resize(frame, (620, 350))
        for i in range(3):
            display_text = f"{decoded_labels[i]},{decoded_top_probs[i] * 100:.2f}%"
            frame = display_text_fnc(frame, display_text, i)

        display_text = f"Infer Time:{processing_time:.1f}ms"
        frame = display_text_fnc(frame, display_text, 3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)

        video_label.after(10, update_frame)  # Schedule next frame

    update_frame()







# # inference.py
# from openvino.runtime import Core
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# import time
# import collections
# import sys
# import os
# from IPython import display

# from core.email_alert import send_email_alert
# from notebook_utils import VideoPlayer #notebook_utils
# #from core.model_loader import compiled_encoder, decoder, output_layer_ir, class_vocab, height_en, frames2decode, IMG_SIZE
# from core.preprocessing import display_text_fnc
# #from constants import frames2decode, height_en

# sys.path.append(os.path.abspath("Proj"))
# # Load the model
# model_path = r"C:/Users/malle/Desktop/Proj/model/classifier_lstm_e19.h5"
# decoder = load_model(model_path, compile=False)

# decoder.summary()

# # Load OpenVINO IR Model
# ie = Core()
# encoder_model_ir = ie.read_model(model=r"C:/Users/malle/Desktop/Proj/inceptionv3_model/saved_model.xml")
# compiled_encoder = ie.compile_model(model=encoder_model_ir, device_name="CPU")

# # Get output layer
# output_layer_ir = compiled_encoder.output(0)

# #Define hyperparameters
# IMG_SIZE = (299, 224)
# BATCH_SIZE = 30
# EPOCHS = 100

# MAX_SEQ_LENGTH = 32
# NUM_FEATURES = 2048

# # Get input shape of the model
# input_shape = compiled_encoder.inputs[0].shape  # Returns a tuple-like OpenVINO Shape object

# # Extract height and width properly
# batch_size, height_en, width_en, channels = input_shape  # Unpack values correctly

# # Ensure 'compiled_encoder' is correctly defined
# if "compiled_encoder" in locals():
#     height_en, width_en = list(compiled_encoder.inputs[0].shape)[1:3]
# else:
#     raise NameError("compiled_encoder is not defined. Ensure the encoder model is loaded correctly.")

# # Ensure 'decoder' is correctly defined
# if "decoder" in locals():
#     frames2decode = list(decoder.inputs[0].shape)[1]  # Fixed incorrect indexing
# else:
#     raise NameError("decoder is not defined. Ensure the LSTM classifier is loaded correctly.")

# # label_processor.get_vocabulary()
# class_vocab = ['Abuse', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting']



# # Used to track frames encoded or not
# frames_idx = []

# def run_action_recognition(
#     source: str = '0',
#     flip: bool = True,
#     skip_first_frames: int = 0
# ):
#     ########################
#     size = height_en  # Encoder input size
#     sample_duration = frames2decode  # Decoder input size
#     fps = 30
#     player = None
#     ########################
#     final_inf_counter = 0
#     final_infer_time = time.time()
#     final_infer_duration = 0
#     #############################################
#     global frames_idx
#     frame_counter = None
#     #############################################
#     try:
#         player = VideoPlayer(source, flip=flip, fps=fps, skip_first_frames=skip_first_frames)
#         player.start()
#         processing_times = collections.deque()
#         processing_time = 0
#         encoder_output = []
#         decoded_labels = [0, 0, 0]
#         decoded_top_probs = [0, 0, 0]
#         counter = 0

#         text_inference_template = "Infer Time:{Time:.1f}ms"
#         text_template = "{label},{conf:.2f}%"

#         while True:
#             counter += 1
#             frame, frame_counter = player.next()

#             if frame is None:
#                 print("Source ended")
#                 break

#             preprocessed = cv2.resize(frame, IMG_SIZE)
#             preprocessed = preprocessed[:, :, [2, 1, 0]]  # BGR -> RGB

#             if counter % 2 == 0:
#                 frames_idx.append((counter, frame_counter, 'Yes'))

#                 start_time = time.time()
#                 encoder_output.append(compiled_encoder([preprocessed[None, ...]])[output_layer_ir][0])

#                 if len(encoder_output) == sample_duration:
#                     frame_mask = np.array([[1]*sample_duration])
#                     encoder_output_np = np.array(encoder_output)[None, ...]
#                     probabilities = decoder.predict(encoder_output_np)[0]

#                     for idx, i in enumerate(np.argsort(probabilities)[::-1][:3]):
#                         decoded_labels[idx] = class_vocab[i]
#                         decoded_top_probs[idx] = probabilities[i]

#                     encoder_output = []
#                     final_inf_counter += 1
#                     final_infer_duration = (time.time() - final_infer_time)
#                     final_infer_time = time.time()

#                     if decoded_labels[0] != "Normal":
#                         send_email_alert(decoded_labels[0], decoded_top_probs[0] * 100)

#                 stop_time = time.time()
#                 processing_times.append(stop_time - start_time)

#                 if len(processing_times) > 200:
#                     processing_times.popleft()

#                 processing_time = np.mean(processing_times) * 1000
#                 fps = 1000 / processing_time

#             else:
#                 frames_idx.append((counter, frame_counter, 'No'))

#             frame = cv2.resize(frame, (620, 350))
#             for i in range(0, 3):
#                 display_text = text_template.format(
#                     label=decoded_labels[i],
#                     conf=decoded_top_probs[i] * 100,
#                 )
#                 frame = display_text_fnc(frame, display_text, i)

#             display_text = text_inference_template.format(Time=processing_time, fps=fps)
#             frame = display_text_fnc(frame, display_text, 3)

#             # _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
#             # i = display.Image(data=encoded_img)
#             # display.clear_output(wait=True)
#             # display.display(i)

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(frame_rgb)
#             img_tk = ImageTk.PhotoImage(image=img)
#             video_label.imgtk = img_tk
#             video_label.configure(image=img_tk)

#             video_label.after(1, update_frame)

#         update_frame()

#     except KeyboardInterrupt:
#         print("Interrupted")
#     except RuntimeError as e:
#         print(e)
#     finally:
#         if player is not None:
#             player.stop()


