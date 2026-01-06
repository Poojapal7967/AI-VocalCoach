import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import time

# --- PREMIUM UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at center, #1a1a3a 0%, #050510 100%); color: #ffffff; }
    .main .block-container { text-align: center; display: flex; flex-direction: column; align-items: center; }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 50px !important; font-weight: 900 !important; }
    [data-testid="stMetricLabel"] { color: #00f2fe !important; font-size: 18px !important; }
    div[data-testid="stMetric"] { 
        background: rgba(255, 255, 255, 0.08) !important; 
        backdrop-filter: blur(20px); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        border-radius: 20px; 
        padding: 20px; 
    }

    /* Improvement Box */
    .improvement-box { background: rgba(255, 75, 75, 0.1); border: 1px solid #ff4b4b; padding: 15px; border-radius: 15px; margin: 10px auto; max-width: 800px; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION INITIALIZATION ---
if 'recording_start' not in st.session_state:
    st.session_state.recording_start = None
if 'raw_audio' not in st.session_state:
    st.session_state.raw_audio = None
if 'analysis_ready' not in st.session_state:
    st.session_state.analysis_ready = False

# AI Model Load
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP HEADER ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("Ab aapka audio utna hi record hoga jitna aap bolenge!")
st.markdown("---")

# Settings
c_lang, c_goal = st.columns(2)
with c_lang: language = st.selectbox("Language", ["English", "Hindi"])
with c_goal: goal = st.selectbox("Goal", ["Public Speaking", "Anchoring", "Teaching", "Interview"])

st.write("##")

# --- CONTROL BUTTONS ---
btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])

# START ACTION
if btn_col1.button("🎤 START", use_container_width=True, type="primary"):
    st.session_state.recording_start = time.time()
    st.session_state.analysis_ready = False
    fs = 44100
    # Ek bada buffer allot kiya (5 minutes)
    st.session_state.raw_audio = sd.rec(int(300 * fs), samplerate=fs, channels=1)
    st.rerun()

# STOP ACTION
if btn_col2.button("🛑 STOP & ANALYZE", use_container_width=True):
    if st.session_state.recording_start:
        # Calculate actual time spent
        end_time = time.time()
        actual_duration = end_time - st.session_state.recording_start
        
        sd.stop() # Mic band kiya
        
        # Audio ko actual duration par crop kiya
        fs = 44100
        cropped_audio = st.session_state.raw_audio[:int(actual_duration * fs)]
        
        # Volume Normalization
        if np.max(np.abs(cropped_audio)) > 0:
            cropped_audio = cropped_audio / np.max(np.abs(cropped_audio))
        
        write('speech.wav', fs, cropped_audio)
        st.session_state.analysis_ready = True
        st.session_state.recording_start = None
        st.rerun()

if btn_col3.button("🔄 RESET", use_container_width=True):
    st.session_state.analysis_ready = False
    st.session_state.recording_start = None
    if os.path.exists('speech.wav'): os.remove('speech.wav')
    st.rerun()

# Recording Status
if st.session_state.recording_start:
    st.warning(f"🔴 Recording in Progress... Bolte rahiye!")

# --- ANALYSIS DISPLAY ---
if st.session_state.analysis_ready and os.path.exists('speech.wav'):
    st.audio('speech.wav')
    
    with st.spinner("⌛ AI Analyzing Performance..."):
        y, sr = librosa.load('speech.wav')
        result = model.transcribe('speech.wav', language=("en" if language=="English" else "hi"))
        text = result['text']
        
        # Accurate Analysis
        actual_time = len(y) / sr
        wpm = int(len(text.split()) / (actual_time / 60)) if actual_time > 0 else 0
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        # Dashboard
        st.markdown(f"### 📊 Analysis Report: {goal}")
        
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#00f2fe')
        ax.set_axis_off()
        fig.patch.set_facecolor('#050510')
        st.pyplot(fig)

        m1, m2, m3 = st.columns(3)
        m1.metric("Time Recorded", f"{round(actual_time, 1)}s")
        m2.metric("Speech Pace", f"{wpm} WPM")
        m3.metric("Voice Tone", "Dynamic" if pitch_var > 60 else "Monotone")

        st.info(f"**Transcription:** {text}")

        # Improvements
        st.subheader("🚀 Improvement Insights")
        if wpm > 165:
            st.markdown('<div class="improvement-box">⚠️ **Tez Bol Rahe Hain:** Thoda dhire bolein taaki log samajh sakein.</div>', unsafe_allow_html=True)
        if len(text.strip()) == 0:
            st.error("❌ AI ko kuch sunai nahi diya. Mic check karein!")
        elif pitch_var < 40:
            st.markdown('<div class="improvement-box">⚠️ **Vocal Variety:** Apni awaaz mein utaar-chadhaw laayein.</div>', unsafe_allow_html=True)
        else:
            st.success("🌟 Great job! Aapki delivery kafi behtar hai.")