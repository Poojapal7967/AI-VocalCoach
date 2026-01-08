import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import os
import time

# --- 1. PREMIUM NEON STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #08081a; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* Neon Glow Cards */
    .attractive-card {
        background: rgba(255, 255, 255, 0.04); border-radius: 20px; padding: 50px 20px; text-align: center;
        backdrop-filter: blur(20px); height: 280px; display: flex; flex-direction: column; justify-content: center;
        transition: 0.5s; position: relative;
    }
    .glow-red { border: 2.5px solid #ff4b4b; box-shadow: 0 0 25px rgba(255, 75, 75, 0.6); }
    .glow-orange { border: 2.5px solid #ff9800; box-shadow: 0 0 25px rgba(255, 152, 0, 0.6); }
    .glow-cyan { border: 2.5px solid #00f2fe; box-shadow: 0 0 25px rgba(0, 242, 254, 0.6); }

    .hero-num { font-size: 85px; font-weight: 900; line-height: 0.8; margin-bottom: 15px; }
    .hero-label { font-size: 14px; color: #a0a0b0; text-transform: uppercase; letter-spacing: 3px; }

    /* Buttons Fix */
    .stButton > button {
        width: 100% !important; border-radius: 50px !important;
        padding: 15px 0px !important; font-weight: 800 !important;
        text-transform: uppercase; transition: 0.3s;
    }
    /* Start Button Red */
    div[data-testid="column"]:nth-of-type(1) .stButton > button {
        border: 2px solid #ff4b4b !important; box-shadow: 0px 0px 30px rgba(255, 75, 75, 0.6);
    }
    /* Stop Button White */
    div[data-testid="column"]:nth-of-type(2) .stButton > button {
        border: 2px solid #ffffff !important; box-shadow: 0px 0px 30px rgba(255, 255, 255, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. BACKEND INITIALIZATION ---
if 'recording' not in st.session_state: st.session_state.recording = False
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False

@st.cache_resource
def load_model(): return whisper.load_model("base")
model = load_model()

# --- 3. PERFORMANCE GRID ---
st.markdown('<p style="color:#ffcc00; font-weight:800; font-size:20px;">📊 PERFORMANCE REPORT</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="attractive-card glow-red"><div class="hero-num">19</div><div class="hero-label">Confidence Score</div></div>', unsafe_allow_html=True)
with col2:
    # Audio Player Box
    st.markdown('<div class="attractive-card glow-orange">', unsafe_allow_html=True)
    if os.path.exists("speech.wav"):
        st.audio("speech.wav")
        st.markdown('<div class="hero-label" style="margin-top:20px;">VOICE RECORDED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#555; font-size:40px;">🔊</div><div class="hero-label">No Record Found</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="attractive-card glow-cyan"><div class="hero-num">3%</div><div class="hero-label">Fillers Found</div></div>', unsafe_allow_html=True)

# --- 4. START & STOP CONTROLS (Logic Fix) ---
st.write("##")
c_l, c_center, c_r = st.columns([1, 2, 1])

with c_center:
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button("🎤 START"):
            st.session_state.recording = True
            st.session_state.analysis_ready = False
            # Sample 5-second recording logic
            fs = 44100
            duration = 5 
            st.toast("Recording for 5 seconds...", icon="🔴")
            rec_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            write("speech.wav", fs, rec_data)
            st.session_state.recording = False
            st.session_state.analysis_ready = True
            st.rerun()

    with btn_col2:
        if st.button("🛑 ANALYZE"):
            if os.path.exists("speech.wav"):
                st.toast("Analyzing with Whisper AI...", icon="⌛")
                result = model.transcribe("speech.wav")
                st.session_state.transcription = result['text']
                st.rerun()
            else:
                st.warning("Pehle recording start karein!")

# --- 5. COACHING FEEDBACK ---
if st.session_state.analysis_ready and 'transcription' in st.session_state:
    st.write("##")
    st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid #00f2fe; border-radius: 20px; padding: 30px;">
            <p style="color:#00f2fe; font-weight:900;">💡 TRANSCRIPTION & TIPS</p>
            <p style="color:#fff; font-style:italic;">"{st.session_state.transcription}"</p>
            <hr style="border-color:#333;">
            <div style="color:#4caf50;">★ Success: Your voice is now visible and analyzed!</div>
        </div>
    """, unsafe_allow_html=True)