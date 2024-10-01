import os
import cv2
import re
import numpy as np
from fpdf import FPDF
from glob import glob
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.models import load_model
import language_tool_python

# Disable OneDNN options for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize LanguageTool for text correction
tool = language_tool_python.LanguageTool('en-US')

# Load the pre-trained CRNN model
model = load_model('files/model.h5', compile=False)

# Vocabulary setup
vocab = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
vocab_size = len(vocab) + 1
max_label_len = 95  # max length of input labels

# Function to correct spelling using LanguageTool
def correct_text(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

# Function to preprocess the image for the CRNN model
def preprocess(img_path):
    image = cv2.imread(img_path)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512, 64))
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = image.astype(np.float32) / 255.0
    return image

# Function to map numbers to characters based on vocabulary
def num_to_label(num):
    text = ''.join([vocab[ch] for ch in num if ch != -1])  # CTC Blank is -1
    return text

# Function to load images and predict text using the CRNN model
def load_and_predict_images(directory):
    image_files = glob(os.path.join(directory, '*.png'))
    image_files.sort(key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', x)])

    test_x = [preprocess(img_file) for img_file in image_files]
    test_x = np.array(test_x).reshape(-1, 512, 64, 1)

    preds = model.predict(test_x)
    decoded = k.get_value(k.ctc_decode(preds, input_length=np.ones(preds.shape[0]) * preds.shape[1], greedy=True)[0][0])

    predicted_text = ''.join(['#' + num_to_label(decoded[i]) for i in range(len(test_x))])
    corrected_text = correct_text(predicted_text).split('#')

    print("Successfully converted to digital text .....")
    return corrected_text

# Function to create a PDF from the predicted text
def create_pdf(data, filename):
    pdf = FPDF(format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=18)

    line_height = pdf.font_size * 2
    for line in data:
        pdf.cell(0, line_height, txt=line, ln=True)

    pdf.output(filename)
    print("Successfully converted to PDF .....")
