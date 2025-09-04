# VoiceCloningAssignment
This repository contains code for converting a PDF book to an audiobook using Deepgram for text-to-speech and OpenVoice for voice cloning.

## Project Overview
- **Objective**: Convert *Life 3.0* by Max Tegmark into an audiobook using my cloned voice.
- **Tools**: Deepgram, OpenVoice, Python.
- **Features**: PDF text extraction, chunking, voice cloning, audio merging, and Google Drive upload.

## Setup
1. Install dependencies: `pip install openvoice pydub torch pdfplumber deepgram-sdk google-auth-oauthlib google-api-python-client`.
2. Place your PDF (e.g., `life_3_0.pdf`) and voice sample (e.g., `my_voice.wav`) in `C:\Users\Admin\Downloads`.
3. Update API keys and paths in `openvoice_pipeline.py` and `deepgram_tts.py`.
4. Run `python voice_cloning.py`.

## Files
- `voice_cloning.ipynb`: Main script for voice cloning and audiobook generation.
- `deepgram.ipynb`: Script for initial TTS using Deepgram.