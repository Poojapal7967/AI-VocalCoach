import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa

# Page UI Styling
st.set_page_config(page_title="AI Vocal Coach", page_icon="🎙️")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #4a4a4a; }
    </style>
    """, unsafe_allow_html=True)

# AI Model Loading
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

st.title("🎙️ AI Vocal Coach Pro")
st.write("Analyze your confidence, tone, and speaking speed.")

duration = st.slider("Recording Time (seconds)", 3, 15, 5)

if st.button("🔴 Start Recording"):
    fs = 44100
    with st.spinner("Recording... Speak clearly!"):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write('speech.wav', fs, recording)
    
    st.audio('speech.wav')

    # AI ANALYSIS SECTION
    with st.spinner("AI is evaluating your performance..."):
        # 1. Transcription & Filler Detection
        result = model.transcribe("speech.wav")
        text = result['text']
        
        # 2. Audio Processing (Pitch & Volume)
        y, sr = librosa.load("speech.wav")
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # 3. Calculation Logic
        words = text.split()
        wpm = int(len(words) / (duration / 60))
        fillers = ["um", "uh", "ah", "like", "hmm", "actually"]
        filler_count = sum(1 for word in words if word.lower().strip(",.") in fillers)
        
        # Confidence Score (Dynamic Formula)
        score = 100 - (filler_count * 10)
        if wpm < 100 or wpm > 180: score -= 20
        score = max(10, score)

    # DISPLAY RESULTS
    st.subheader("📊 Performance Report")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence Score", f"{score}%")
    col2.metric("Fillers Found", filler_count)
    col3.metric("Speaking Speed", f"{wpm} WPM")

    st.markdown(f"**What you said:** *\"{text}\"*")

    # Final Expert Advice
    st.divider()
    st.subheader("💡 Coaching Tips")
    
    if score > 80:
        st.success("🌟 Excellent! Your speech is clear and confident.")
    elif score > 50:
        st.warning("⚡ Good effort, but there's room for improvement.")
    else:
        st.error("📉 Focus on your flow. Try to minimize pauses and filler words.")

    if wpm < 110:
        st.info("🐢 **Speed:** You are speaking a bit slow. Try to be more fluent.")
    elif wpm > 170:
        st.info("🚀 **Speed:** You are speaking too fast. Slow down to let the audience breathe.")