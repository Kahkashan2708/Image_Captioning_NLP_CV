import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from PIL import Image

# ---------- Page Config ----------
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Load Models & Tokenizer (cached for efficiency) ----------
@st.cache_resource
def load_resources(model_path, feature_extractor_path, tokenizer_path):
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer

# ---------- Generate Caption ----------
def generate_caption(image, caption_model, feature_extractor, tokenizer, max_length=34, img_size=224):
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
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption

# ---------- Streamlit App ----------
def main():
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.info("Upload an image and generate captions using your trained model.")
    max_length = st.sidebar.slider("Max Caption Length", 20, 50, 34)

    st.markdown(
        """
        <style>
        .title {
            font-size:36px !important;
            font-weight:bold;
            color:#2E86C1;
            text-align:center;
        }
        .caption-box {
            padding:20px;
            background-color:#F2F4F4;
            border-radius:12px;
            margin-top:20px;
            font-size:26px; /* Larger caption text */
            font-weight:bold;
            text-align:center;
            color:#1C2833;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown('<p class="title"> Image Caption Generator</p>', unsafe_allow_html=True)
    st.write("Upload an image to generate a meaningful caption using trained model.")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open image with PIL
        image = Image.open(uploaded_image).convert("RGB")

        # Paths for models/tokenizer (adjust to your files)
        model_path = "model.keras"
        tokenizer_path = "tokenizer.pkl"
        feature_extractor_path = "feature_extractor.keras"

        # Load models and tokenizer
        caption_model, feature_extractor, tokenizer = load_resources(
            model_path, feature_extractor_path, tokenizer_path
        )

        with st.spinner("⏳ Generating caption..."):
            caption = generate_caption(image, caption_model, feature_extractor, tokenizer, max_length=max_length)

        # Layout: Two columns (image smaller, caption larger)
        col1, col2 = st.columns([1, 2])  # Image takes 2/3, caption 1/3

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)  

        with col2:
            st.markdown('<div class="caption-box"> ' + caption + '</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()



