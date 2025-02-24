import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# 1) Load your .h5 file (Adjust the path to your actual file location)
MODEL_PATH = r"C:\Users\henry\Desktop\Personal\Training\Project\Personal-Project-on-X-Ray-Image-Classifier-Pneumonia-Detection\notebook\tuned_resnet50_model.tflite"
model = tf.keras.models.load_model(MODEL_PATH)

# 2) (Optional) Force a dummy forward pass for Keras to define the input
_ = model.predict(np.zeros((1, 64, 64, 3), dtype=np.float32))

# Class labels for a binary Normal/Pneumonia classification
class_labels = ["Normal", "Pneumonia"]

def preprocess_image(img):
    """
    1. Resize the image to (64, 64).
    2. Normalize pixel values to [0, 1].
    3. Expand dimension to shape (1, 64, 64, 3).
    """
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def make_prediction(img):
    """
    1. Preprocess the image.
    2. Predict with the loaded model.
    3. Return the predicted label ('Normal' or 'Pneumonia') and confidence.
    """
    x = preprocess_image(img)
    pred = model.predict(x)[0][0]  # Single scalar output
    confidence = float(pred) if pred > 0.5 else float(1 - pred)
    label = class_labels[int(pred > 0.5)]
    return label, confidence

# ---------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------
st.title("Pneumonia Detector (Minimal)")

# Upload an X-ray image
uploaded_file = st.file_uploader("Upload a chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file into a NumPy array (OpenCV format)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR format
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    # Convert to RGB

    # Display the uploaded image
    st.image(img_rgb, caption="Uploaded X-ray Image", use_container_width=True)

    # 1) Make a prediction
    label, confidence = make_prediction(img_rgb)

    # 2) Display the result
    st.write(f"### Prediction: {label} ({confidence*100:.2f}%)")
