import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from PIL import Image

# ---------- Load Models & Tokenizer (cached for efficiency) ----------
def load_resources():
    model_path = "model.keras"
    feature_extractor_path = "feature_extractor.keras"
    tokenizer_path = "tokenizer.pkl"

    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer


caption_model, feature_extractor, tokenizer = load_resources()


# ---------- Generate Caption ----------
def generate_caption(image, max_length=34, img_size=224):
    # Convert numpy array (Gradio gives) to PIL
    img = Image.fromarray(image.astype("uint8")).convert("RGB")

    # Preprocess image
    img = img.resize((img_size, img_size))
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


# ---------- Gradio App ----------
def inference(image, max_length):
    caption = generate_caption(image, max_length=max_length)
    return caption


# Build Gradio Interface
demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(type="numpy", label="Upload an Image"),
        gr.Slider(20, 50, value=34, step=1, label="Max Caption Length"),
    ],
    outputs=gr.Textbox(label="Generated Caption"),
    title=" Image Caption Generator",
    description="Upload an image and generate captions using your trained model.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()




