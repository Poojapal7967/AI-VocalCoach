import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# --- UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at center, #1a1a3a 0%, #050510 100%); color: #ffffff; }
    .main .block-container { text-align: center; display: flex; flex-direction: column; align-items: center; }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 50px !important; font-weight: 900 !important; display: flex; justify-content: center; }
    [data-testid="stMetricLabel"] { color: #00f2fe !important; font-size: 18px !important; display: flex; justify-content: center; width: 100%; }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.08) !important; backdrop-filter: blur(20px) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 25px !important; padding: 30px !important; text-align: center; }
    
    /* Improvement Card */
    .improvement-box { background: rgba(255, 75, 75, 0.1); border: 1px solid #ff4b4b; padding: 20px; border-radius: 15px; margin: 10px auto; max-width: 800px; text-align: center; }
    
    /* Button Styles */
    .stButton>button {
        width: 100% !important; border-radius: 50px !important;
        padding: 15px !important; font-size: 20px !important; font-weight: bold !important;
        border: none !important; color: white !important;
    }
    .start-btn { background: linear-gradient(90deg, #ff4b4b, #ff9068) !important; }
    .stop-btn { background: linear-gradient(90deg, #7000ff, #00f2fe) !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'last_audio' not in st.session_state:
    st.session_state.last_audio = None

# Model Load
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP UI ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("Manual Start/Stop aur Fixed Recording Player ke saath")
st.markdown("---")

# Settings
c1, c2 = st.columns(2)
with c1: language = st.selectbox("Language", ["English", "Hindi"])
with c2: goal = st.selectbox("Goal", ["Public Speaking", "Anchoring", "Teaching", "Interview"])

st.write("##")

# Callback function to collect audio data
def audio_callback(indata, frames, time, status):
    if st.session_state.recording:
        st.session_state.audio_buffer.append(indata.copy())

# --- MANUAL CONTROL BUTTONS ---
btn_col1, btn_col2, btn_col3 = st.columns([1,2,1])

with btn_col2:
    if not st.session_state.recording:
        if st.button("🎤 START RECORDING", type="primary"):
            st.session_state.recording = True
            st.session_state.audio_buffer = []
            st.session_state.last_audio = None
            st.rerun()
    else:
        st.warning("🔴 Recording in Progress... Bolte rahiye!")
        if st.button("🛑 STOP & ANALYZE"):
            st.session_state.recording = False
            if st.session_state.audio_buffer:
                # Combine buffer
                audio_data = np.concatenate(st.session_state.audio_buffer, axis=0)
                # Normalize & Boost
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = (audio_data / np.max(np.abs(audio_data))) * 0.9
                write('speech.wav', 44100, audio_data)
                st.session_state.last_audio = 'speech.wav'
            st.rerun()

# Recording context handler
if st.session_state.recording:
    with sd.InputStream(samplerate=44100, channels=1, callback=audio_callback):
        while st.session_state.recording:
            import time
            time.sleep(0.1)

# --- RESULTS SECTION ---
if st.session_state.last_audio and os.path.exists(st.session_state.last_audio):
    st.markdown("### 🎧 Your Recording")
    st.audio(st.session_state.last_audio)
    
    with st.spinner("⌛ AI Analysis in progress..."):
        # Whisper Transcription
        result = model.transcribe(st.session_state.last_audio, language=("en" if language=="English" else "hi"))
        text = result['text']
        y, sr = librosa.load(st.session_state.last_audio)
        
        # Performance Metrics
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        duration_sec = len(y) / sr
        wpm = int(len(text.split()) / (duration_sec / 60)) if duration_sec > 0 else 0
        
        # Filler word logic
        fillers = ["um", "uh", "ah", "like", "hmm", "matlab", "toh", "basically"]
        filler_count = sum(1 for word in text.split() if word.lower().strip(",.") in fillers)

        # Dashboard Display
        st.markdown(f"### 📊 Analysis for {goal}")
        
        # Centered Waveform
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#00f2fe')
        ax.set_axis_off()
        fig.patch.set_facecolor('#050510')
        st.pyplot(fig)

        m1, m2, m3 = st.columns(3)
        m1.metric("Speech Pace", f"{wpm} WPM")
        m2.metric("Filler Words", filler_count)
        m3.metric("Voice Energy", "High" if pitch_var > 70 else "Low")

        st.info(f"**Transcription:** {text}")

        # Improvements logic
        st.subheader("🚀 Improvement Insights")
        improvements = []
        if wpm > 165: improvements.append("⚠️ **Too Fast:** Please speak a bit slower for better impact.")
        if filler_count > 2: improvements.append(f"⚠️ **Fillers:** You used {filler_count} fillers. Try to replace them with pauses.")
        if pitch_var < 40: improvements.append("⚠️ **Monotone Voice:** Try adding more pitch variation to sound more engaging.")

        if not improvements:
            st.success("🌟 Great Job! Your delivery was smooth and professional.")
        else:
            for imp in improvements:
                st.markdown(f'<div class="improvement-box">{imp}</div>', unsafe_allow_html=True)