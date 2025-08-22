import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from PIL import Image
import base64
import os

# ---------- Paths to models & tokenizer ----------
# Assumes model files are one level up from /src/
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model.keras"))
FEATURE_EXTRACTOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../feature_extractor.keras"))
TOKENIZER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokenizer.pkl"))

# ---------- Streamlit Page Config ----------
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Cache Resources ----------
@st.cache_resource
def load_resources(model_path, feature_extractor_path, tokenizer_path):
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer

# ---------- Caption Generation ----------
def generate_caption(image, caption_model, feature_extractor, tokenizer, max_length=34, img_size=224, method="Greedy"):
    # Preprocess image
    img = image.resize((img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Extract image features
    image_features = feature_extractor.predict(img_array, verbose=0)

    # Generate caption
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)

        if method == "Greedy":
            yhat_index = np.argmax(yhat)
        elif method == "Sampling":
            yhat_index = np.random.choice(len(yhat[0]), p=yhat[0])
        else:
            yhat_index = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption

# ---------- Download Helper ----------
def get_download_link(text, filename="caption.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Caption</a>'

# ---------- Main App ----------
def main():
    # Sidebar settings
    st.sidebar.title("⚙️ Settings")
    st.sidebar.info("Upload an image and generate captions using the trained model.")
    max_length = st.sidebar.slider("Max Caption Length", 20, 50, 34)
    method = st.sidebar.radio("Generation Method", ["Greedy", "Sampling"])
    theme = st.sidebar.radio("Theme Mode", ["🌞 Light", "🌙 Dark"])

    # Theme customization
    if theme == "🌞 Light":
        bg_color = "#F2F4F4"
        text_color = "#1C2833"
    else:
        bg_color = "#1C2833"
        text_color = "#F2F4F4"

    st.markdown(
        f"""
        <style>
        .title {{
            font-size:36px !important;
            font-weight:bold;
            color:#2E86C1;
            text-align:center;
        }}
        .caption-box {{
            padding:20px;
            background-color:{bg_color};
            border-radius:12px;
            margin-top:20px;
            font-size:26px;
            font-weight:bold;
            text-align:center;
            color:{text_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and Instructions
    st.markdown('<p class="title"> 📷 Image Caption Generator</p>', unsafe_allow_html=True)
    st.write("Upload an image to generate a meaningful caption using your trained model.")

    # File uploader
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")

        # Load resources
        caption_model, feature_extractor, tokenizer = load_resources(
            MODEL_PATH, FEATURE_EXTRACTOR_PATH, TOKENIZER_PATH
        )

        with st.spinner("⏳ Generating caption..."):
            caption = generate_caption(image, caption_model, feature_extractor, tokenizer,
                                       max_length=max_length, method=method)

        # Layout: image + caption
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.markdown(f'<div class="caption-box">{caption}</div>', unsafe_allow_html=True)

        # Optional edit and download
        new_caption = st.text_input("✏️ Edit your caption:", caption)
        if new_caption != caption:
            caption = new_caption

        st.markdown(get_download_link(caption), unsafe_allow_html=True)
        st.button("📋 Copy to Clipboard")

if __name__ == "__main__":
    main()
