import streamlit as st
import openai
import tempfile
import os
import requests
from PIL import Image
from io import BytesIO

# Load API Key from Environment Variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit App Title
st.title("🎨 AI-Powered Image Modification with Speech Instructions")

# Upload Image
uploaded_image = st.file_uploader("📤 Upload an image:", type=["png", "jpg", "jpeg"])

# Upload Audio File
st.write("🎙️ Upload your speech instruction for image modification.")
audio_file = st.file_uploader("📥 Upload an audio file (WAV/MP3):", type=["wav", "mp3"])

if uploaded_image and audio_file:
    # Save the audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name  # Store file path safely

    try:
        # Transcribe Speech using OpenAI Whisper (Latest API)
        with open(temp_audio_path, "rb") as f:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

        transcribed_text = transcription.text
        st.write("📝 Transcribed Instruction: ", transcribed_text)

        # Modify the Image Using DALL-E 3
        response = openai.images.generate(
            prompt=transcribed_text,
            model="dall-e-3",
            n=1,
            size="1024x1024"
        )

        modified_image_url = response.data[0].url
        st.image(modified_image_url, caption="🖼️ Modified Image", use_column_width=True)

        # Save the original and modified images
        original_image = Image.open(uploaded_image)
        modified_image = Image.open(BytesIO(requests.get(modified_image_url).content))

        original_image_path = "uploaded_image.jpg"
        modified_image_path = "modified_image.jpg"

        original_image.save(original_image_path)
        modified_image.save(modified_image_path)

        st.success("✅ Image modification complete!")

    except Exception as e:
        st.error(f"❌ Error: {e}")

    finally:
        # Cleanup Temporary Files
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
