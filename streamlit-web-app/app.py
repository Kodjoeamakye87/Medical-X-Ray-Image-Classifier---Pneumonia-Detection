import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# ---------------------------------------------------------
# Load TensorFlow Lite Model
# ---------------------------------------------------------
MODEL_PATH = r"C:\Users\henry\Desktop\Personal\Training\Project\Personal-Project-on-X-Ray-Image-Classifier-Pneumonia-Detection\Medical-X-Ray-Image-Classifier---Pneumonia-Detection\notebook\tuned_resnet50_model.tflite"

# Initialize TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels for binary classification
class_labels = ["Normal", "Pneumonia"]

# ---------------------------------------------------------
# Preprocessing Function
# ---------------------------------------------------------
def preprocess_image(img):
    """
    1. Resize the image to match model input.
    2. Normalize pixel values to [0, 1].
    3. Expand dimensions to fit model input shape.
    """
    img = cv2.resize(img, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------
def make_prediction(img):
    """
    1. Preprocess the image.
    2. Run inference with TensorFlow Lite.
    3. Return predicted label and confidence score.
    """
    x = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]  # Single scalar output
    confidence = float(pred) if pred > 0.5 else float(1 - pred)
    label = class_labels[int(pred > 0.5)]
    return label, confidence

# ---------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------
st.title("Pneumonia Detector (Medical X-Ray Image Classifier)")
st.write("Upload a chest X-ray image to detect Pneumonia.")

# Upload an X-ray image
uploaded_file = st.file_uploader("Upload a chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file into a NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Load image in BGR format
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    # Convert to RGB

    # Display the uploaded image
    st.image(img_rgb, caption="Uploaded X-ray Image", use_container_width=True)

    # 1) Make a prediction
    label, confidence = make_prediction(img_rgb)

    # 2) Display the result
    st.write(f"### Prediction: {label} ({confidence * 100:.2f}%)")
