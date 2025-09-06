import os
import re
import glob
import torch
import logging
import numpy as np
import pdfplumber
import streamlit as st
from datetime import datetime
from pydub import AudioSegment
from TTS.api import TTS
from scipy.io import wavfile
from librosa import resample
from deepgram import DeepgramClient, SpeakOptions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Life 3.0 Audiobook Generator")
st.write("Upload a PDF and voice sample to create an audiobook in your voice!")

# File uploads
pdf_file = st.file_uploader("Upload Life 3.0 PDF", type="pdf")
voice_file = st.file_uploader("Upload Voice Sample (.wav or .mp3)", type=["wav", "mp3"])
page_range = st.text_input("Page Range (e.g., 1-10, leave blank for full book)", "")
DEEPGRAM_API_KEY = st.text_input("Enter Deepgram API Key", type="password")
use_existing = st.checkbox("Use existing audiobook files if available", value=True)

if st.button("Generate Audiobook"):
    if not pdf_file or not voice_file or not DEEPGRAM_API_KEY:
        st.error("Please upload PDF, voice sample, and enter Deepgram API key.")
    else:
        # Use relative paths for deployment
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(base_dir, "life_3_0.pdf")
        voice_path = os.path.join(base_dir, "my_voice.wav")
        deepgram_wav = os.path.join(output_dir, f"life_30_deepgram_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        mp3_path = os.path.join(output_dir, f"life_30_yourtts_cloned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")

        # Save uploaded PDF
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        # Always convert uploaded voice to WAV
        raw_voice_path = os.path.join(base_dir, "uploaded_voice")
        with open(raw_voice_path, "wb") as f:
            f.write(voice_file.read())

        try:
            if voice_file.name.endswith(".mp3"):
                audio = AudioSegment.from_mp3(raw_voice_path)
            else:
                audio = AudioSegment.from_wav(raw_voice_path)

            audio = audio.set_channels(1).set_frame_rate(22050)
            voice_path = os.path.join(base_dir, "my_voice.wav")
            audio.export(voice_path, format="wav")
        except Exception as e:
            st.error(f"Error converting uploaded voice to WAV: {str(e)}")
            st.stop()

        # Extract text
        def extract_text(pdf_path, page_range=None):
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                start, end = 0, len(pdf.pages)
                if page_range and page_range.strip():
                    try:
                        s, e = page_range.split('-')
                        start, end = int(s) - 1, int(e)
                    except ValueError:
                        st.error("Invalid page range format. Use 'start-end' (e.g., '1-10').")
                        return None
                for page in pdf.pages[start:end]:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
            return text

        try:
            book_text = extract_text(pdf_path, page_range)
            if book_text is None:
                st.error("Text extraction failed. Please check page range.")
                st.stop()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            st.stop()

        st.write(f"Extracted {len(book_text)} characters from book.")
        with open(os.path.join(base_dir, 'book_text.txt'), 'w') as f:
            f.write(book_text)

        # Deepgram TTS
        existing_deepgram = glob.glob(os.path.join(output_dir, "life_30_deepgram_tts_*.wav"))
        if use_existing and existing_deepgram:
            deepgram_wav = max(existing_deepgram, key=os.path.getmtime)
            st.write(f"Using existing Deepgram TTS: {deepgram_wav}")
        else:
            deepgram = DeepgramClient(DEEPGRAM_API_KEY)
            chunks = [book_text[i:i+2000] for i in range(0, len(book_text), 2000)]
            deepgram_chunks = []
            options = SpeakOptions(model="aura-luna-en", encoding="linear16", container="wav")

            for idx, chunk in enumerate(chunks):
                st.write(f"Processing Deepgram chunk {idx+1}/{len(chunks)}...")
                try:
                    logger.info(f"Chunk {idx+1} length: {len(chunk)} characters")
                    temp_path = os.path.join(base_dir, f"deepgram_chunk_{idx}.wav")
                    deepgram.speak.v("1").save(temp_path, {"text": chunk}, options)
                    deepgram_chunks.append(temp_path)
                except Exception as e:
                    logger.error(f"Deepgram error on chunk {idx+1}: {str(e)}")
                    st.error(f"Deepgram error on chunk {idx+1}: {str(e)}")
                    break

            if deepgram_chunks:
                combined_deepgram = AudioSegment.empty()
                for path in deepgram_chunks:
                    combined_deepgram += AudioSegment.from_wav(path)
                    os.remove(path)
                combined_deepgram.export(deepgram_wav, format="wav")
                st.write(f"Deepgram TTS saved to: {deepgram_wav}")

        if os.path.exists(deepgram
