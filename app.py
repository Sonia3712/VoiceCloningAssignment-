
import streamlit as st
import pdfplumber
import os
import re
import torch
from pydub import AudioSegment
from TTS.api import TTS
from scipy.io import wavfile
import numpy as np
from librosa import resample
from deepgram import DeepgramClient, SpeakOptions
from datetime import datetime
import glob
import logging

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
use_existing = st.checkbox("Use existing audiobook files from Google Drive if available", value=True)

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

        if os.path.exists(deepgram_wav):
            st.audio(deepgram_wav)
            with open(deepgram_wav, "rb") as f:
                st.download_button("Download Deepgram TTS", f.read(), file_name=os.path.basename(deepgram_wav))

        # YourTTS Voice Cloning
        existing_yourtts = glob.glob(os.path.join(output_dir, "life_30_yourtts_cloned_*.mp3"))
        if use_existing and existing_yourtts:
            mp3_path = max(existing_yourtts, key=os.path.getmtime)
            st.write(f"Using existing YourTTS cloned audiobook: {mp3_path}")
        else:
            # Now safely read as WAV
            try:
                samplerate, data = wavfile.read(voice_path)
                if data.dtype != np.float32:
                    data = data.astype(np.float32) / np.iinfo(data.dtype).max
                if samplerate != 22050:
                    data = resample(data, orig_sr=samplerate, target_sr=22050)
                    resampled_path = os.path.join(base_dir, "resampled_voice.wav")
                    wavfile.write(resampled_path, 22050, (data * 32767).astype(np.int16))
                    voice_path = resampled_path
            except Exception as e:
                st.error(f"Error reading WAV file: {str(e)}")
                st.stop()

            try:
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
                st.write("YourTTS model loaded.")
            except Exception as e:
                st.error(f"Failed to load YourTTS: {str(e)}")
                st.stop()

            chunk_size = 2000
            chunks = [book_text[i:i+2000] for i in range(0, len(book_text), 2000)]
            output_files = []

            for i, chunk in enumerate(chunks):
                st.write(f"Generating cloned chunk {i+1}/{len(chunks)}...")
                output_path = os.path.join(output_dir, f"chunk_{i}.wav")
                try:
                    tts.tts_to_file(text=chunk, file_path=output_path, speaker_wav=voice_path, language='en')
                    output_files.append(output_path)
                except Exception as e:
                    st.error(f"Voice cloning error on chunk {i+1}: {str(e)}")
                    break

            if output_files:
                combined = AudioSegment.empty()
                for output_path in output_files:
                    combined += AudioSegment.from_wav(output_path)
                combined.export(mp3_path, format="mp3")
                st.write(f"Cloned audiobook saved to: {mp3_path}")

        if os.path.exists(mp3_path):
            st.audio(mp3_path)
            with open(mp3_path, "rb") as f:
                st.download_button("Download Cloned Audiobook", f.read(), file_name=os.path.basename(mp3_path))
