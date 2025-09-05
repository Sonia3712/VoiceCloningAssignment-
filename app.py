
import streamlit as st
import pdfplumber
import os
import re
import torch
import torchaudio
from pydub import AudioSegment
from TTS.api import TTS
from scipy.io import wavfile
import numpy as np
from librosa import resample
from deepgram import DeepgramClient, SpeakOptions
from datetime import datetime
import glob

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
        # Save uploaded files
        pdf_path = "/content/life_3_0.pdf"
        voice_path = "/content/my_voice.wav"
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())
        with open(voice_path, "wb") as f:
            f.write(voice_file.read())

        # Generate unique filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        deepgram_wav = f"/content/drive/My Drive/life_30_deepgram_tts_20250905_165745.wav"
        mp3_path = f"/content/drive/My Drive/life_30_yourtts_cloned_20250905_165745.mp3"

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
        with open('/content/book_text.txt', 'w') as f:
            f.write(book_text)

        # Deepgram TTS
        existing_deepgram = glob.glob("/content/drive/My Drive/life_30_deepgram_tts_*.wav")
        if use_existing and existing_deepgram:
            deepgram_wav = max(existing_deepgram, key=os.path.getmtime)
            st.write(f"Using existing Deepgram TTS: {deepgram_wav}")
        else:
            deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
            chunks = [book_text[i:i+2000] for i in range(0, len(book_text), 2000)]
            deepgram_chunks = []
            for idx, chunk in enumerate(chunks):
                st.write(f"Processing Deepgram chunk {idx+1}/{len(chunks)}...")
                try:
                    # Debug: Log chunk content
                    st.write(f"Chunk {idx+1} length: {len(chunk)} characters")
                    payload = {"text": chunk}
                    options = SpeakOptions(model="aura-stella-en")
                    response = deepgram.speak.v1.speak(payload, options)
                    temp_path = f"/content/deepgram_chunk_{idx}.wav"
                    with open(temp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    deepgram_chunks.append(temp_path)
                except Exception as e:
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
        existing_yourtts = glob.glob("/content/drive/My Drive/life_30_yourtts_cloned_*.mp3")
        if use_existing and existing_yourtts:
            mp3_path = max(existing_yourtts, key=os.path.getmtime)
            st.write(f"Using existing YourTTS cloned audiobook: {mp3_path}")
        else:
            if voice_path.endswith('.mp3'):
                try:
                    audio = AudioSegment.from_file(voice_path, format="mp3")
                    audio = audio.set_channels(1).set_frame_rate(22050)
                    voice_path = "/content/my_voice.wav"
                    audio.export(voice_path, format="wav", codec="pcm_s16le")
                except Exception as e:
                    st.error(f"Error converting MP3 to WAV: {str(e)}")
                    st.stop()

            try:
                samplerate, data = wavfile.read(voice_path)
                if data.dtype != np.float32:
                    data = data.astype(np.float32) / np.iinfo(data.dtype).max
                if samplerate != 22050:
                    data = resample(data, orig_sr=samplerate, target_sr=22050)
                    wavfile.write("/content/resampled_voice.wav", 22050, (data * 32767).astype(np.int16))
                    voice_path = "/content/resampled_voice.wav"
            except Exception as e:
                st.error(f"Error reading WAV file: {str(e)}")
                st.stop()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
                tts.to(device)
                st.write("YourTTS model loaded.")
            except Exception as e:
                st.error(f"Failed to load YourTTS: {str(e)}")
                st.stop()

            output_dir = "/content/life_30_chunks/"
            os.makedirs(output_dir, exist_ok=True)
            chunk_size = 2000
            chunks = [book_text[i:i+2000] for i in range(0, len(book_text), 2000)]
            output_files = []

            for i, chunk in enumerate(chunks):
                st.write(f"Generating cloned chunk {i+1}/{len(chunks)}...")
                output_path = f"{output_dir}/chunk_{i}.wav"
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
