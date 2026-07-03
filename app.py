import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import streamlit as st

# 1. Page Config
st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶")
st.title("🐶 Dog Breed Classifier")
st.write("Upload a dog photo to predict its breed using EfficientNetV2B3.")

# 2. Cached Loaders
@st.cache_resource
def load_onnx_model():
    # Load ONNX model (< 50MB RAM footprint)
    session = ort.InferenceSession("models/efficientnetv2b3_dog_breed.onnx")
    return session

@st.cache_data
def load_class_names():
    # Properly parse JSON array into a Python list
    with open("models/class_names.json", "r") as f:
        class_names = json.load(f)
    return class_names

session = load_onnx_model()
CLASS_NAMES = load_class_names()

# 3. File Upload Interface
uploaded_file = st.file_uploader("Upload a dog photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)
    
    # Preprocess: EfficientNetV2 expects float32 images in [0, 255] range
    img_resized = image.resize((300, 300))
    img_array = np.array(img_resized, dtype=np.float32)  # Do NOT divide by 255.0
    img_batch = np.expand_dims(img_array, axis=0)        # Shape: (1, 300, 300, 3)
    
    # 4. Inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    raw_output = session.run([output_name], {input_name: img_batch})[0]
    predictions = np.squeeze(raw_output)  # Squeeze to 1D vector (120,)

    # Determine if output is raw logits or already Softmax probabilities
    if np.isclose(np.sum(predictions), 1.0, atol=1e-2):
        probabilities = predictions
    else:
        # Numerically stable Softmax for logits
        exp_preds = np.exp(predictions - np.max(predictions))
        probabilities = exp_preds / exp_preds.sum()
    
    # 5. Extract Top Prediction & Display
    top_idx = int(np.argmax(probabilities))
    top_breed = CLASS_NAMES[top_idx]
    confidence = probabilities[top_idx] * 100

    st.success(f"**Prediction:** {top_breed}")
    st.metric(label="Confidence", value=f"{confidence:.2f}%")
    st.progress(min(float(probabilities[top_idx]), 1.0))

    # Show Top 3 Predictions
    with st.expander("See Top 3 Predictions"):
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        for idx in top_3_indices:
            st.write(f"**{CLASS_NAMES[idx]}:** {probabilities[idx] * 100:.2f}%")