from openvino.runtime import Core
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import time
import collections
import sys
import os


sys.path.append(os.path.abspath("Proj"))
# Load the model
model_path = r"C:/Users/malle/Desktop/Proj/model/classifier_lstm_e19.h5"
decoder = load_model(model_path, compile=False)

decoder.summary()

# Load OpenVINO IR Model
ie = Core()
encoder_model_ir = ie.read_model(model=r"C:/Users/malle/Desktop/Proj/inceptionv3_model/saved_model.xml")
compiled_encoder = ie.compile_model(model=encoder_model_ir, device_name="CPU")

# Get output layer
output_layer_ir = compiled_encoder.output(0)

#Define hyperparameters
IMG_SIZE = (299, 224)
BATCH_SIZE = 30
EPOCHS = 100

MAX_SEQ_LENGTH = 32
NUM_FEATURES = 2048

# Get input shape of the model
input_shape = compiled_encoder.inputs[0].shape  # Returns a tuple-like OpenVINO Shape object

# Extract height and width properly
batch_size, height_en, width_en, channels = input_shape  # Unpack values correctly

# Ensure 'compiled_encoder' is correctly defined
if "compiled_encoder" in locals():
    height_en, width_en = list(compiled_encoder.inputs[0].shape)[1:3]
else:
    raise NameError("compiled_encoder is not defined. Ensure the encoder model is loaded correctly.")

# Ensure 'decoder' is correctly defined
if "decoder" in locals():
    frames2decode = list(decoder.inputs[0].shape)[1]  # Fixed incorrect indexing
else:
    raise NameError("decoder is not defined. Ensure the LSTM classifier is loaded correctly.")

# label_processor.get_vocabulary()
class_vocab = ['Abuse', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting']