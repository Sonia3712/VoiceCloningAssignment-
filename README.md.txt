Life 3.0 Audiobook Generator
Overview
This project converts Max Tegmark’s Life 3.0 PDF into an audiobook using Deepgram for text-to-speech and Coqui TTS (YourTTS) for voice cloning. A Streamlit app provides a user-friendly interface to upload a PDF and voice sample, generating both a Deepgram-based WAV and a cloned MP3 audiobook. Outputs are saved to Google Drive.
Features

Text Extraction: Extracts text from Life 3.0 PDF using pdfplumber.
Deepgram TTS: Generates a natural-sounding audiobook with Deepgram’s aura-asteria-en model.
Voice Cloning: Uses YourTTS to clone the user’s voice from a sample (.mp3 or .wav).
Streamlit App: Interactive interface for file uploads and audiobook generation.
Google Drive Integration: Saves outputs to /content/drive/My Drive/.

Prerequisites

Google Colab with GPU (optional, CPU works but slower).
Deepgram API key (sign up at deepgram.com).
Google Drive access for saving outputs.
Python 3.10 for TTS installation.

Setup

Clone the Repository:
git clone https://github.com/Sonia3712/VoiceCloningAssignment-.git
cd VoiceCloningAssignment


Install Dependencies:Run the following in a Google Colab notebook:
!apt-get update -qq
!apt-get install -y software-properties-common
!add-apt-repository ppa:deadsnakes/ppa -y
!apt-get update -qq
!apt-get install -y python3.10 python3.10-dev python3.10-distutils
!ln -sf /usr/bin/python3.10 /usr/local/bin/python3.10
!curl -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
!python3.10 /tmp/get-pip.py
!python3.10 -m venv /content/myenv
!source /content/myenv/bin/activate; pip install --upgrade pip -q
!source /content/myenv/bin/activate; pip cache purge
!source /content/myenv/bin/activate; pip install --no-cache-dir --index-url https://pypi.org/simple TTS==0.14.3 numpy scipy torch torchaudio pydub pdfplumber librosa -q
!pip install --upgrade pip -q
!pip install --no-cache-dir deepgram-sdk pydub pdfplumber torch torchaudio -q


Prepare Files:

Upload life_3_0.pdf (or your Life 3.0 PDF).
Upload my_voice.wav or my_voice.mp3 (10–30 seconds, clear audio).


Run the Notebook:Execute the cells in order:

Cell 1: Install dependencies.
Cell 2: Mount Google Drive and upload files.
Cell 3: Extract text from PDF.
Cell 4: Generate Deepgram TTS audiobook.
Cell 5: Generate cloned audiobook with YourTTS.






Usage

Notebook:
Run cells sequentially in Colab.
Outputs: /content/drive/My Drive/life_30_deepgram_tts.wav and /content/drive/My Drive/life_30_yourtts_cloned.mp3.


Streamlit App:
Upload PDF and voice sample.
Enter Deepgram API key.
Specify page range (optional).
Click “Generate Audiobook” to create and preview outputs.



Files

cell_1_setup.py: Installs dependencies.
cell_2_upload.py: Mounts Google Drive and uploads files.
cell_3_extract_text.py: Extracts and cleans text.
cell_4_deepgram_tts.py: Generates Deepgram audiobook.
cell_5_yourtts_clone.py: Generates cloned audiobook.
app.py: Streamlit app for interactive use.

Notes

Text Cleaning: Regex includes numbers for titles like “Life 3.0”.
Chunk Size: 2000 characters for stability.
Voice Sample: Ensure mono, 22050 Hz for YourTTS.
Performance: CPU processing is slow; use GPU if available.
Debugging: Check Python version (!/content/myenv/bin/python --version) and dependencies (!source /content/myenv/bin/activate; pip list).

Links

GitHub: https://github.com/Sonia3712/VoiceCloningAssignment-.git

Medium Blog: https://github.com/Sonia3712/VoiceCloningAssignment-.git
LinkedIn: https://www.linkedin.com/posts/sonia-j-3670332b2_ai-machinelearning-voicecloning-activity-7369282794277044227-H8Bi?utm_source=share&utm_medium=member_android&rcm=ACoAAEscaT8BHEOJGAY-sAWGXOh9lKddiuWtW_U

