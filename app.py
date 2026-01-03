import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

st.set_page_config(page_title="AI Vocal Coach", layout="centered")

st.title("🎙️ AI Vocal Coach")
st.subheader("Apni voice record karein aur AI se feedback lein")

duration = st.slider("Kitne seconds record karna hai?", 3, 10, 5)

if st.button("🔴 Record Karein"):
    fs = 44100  # Sample rate
    with st.spinner("Recording..."):
        # Recording start
        my_recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        write('my_voice.wav', fs, my_recording) # Save as wav file
    st.success("Recording save ho gayi: 'my_voice.wav'")
    st.audio('my_voice.wav')