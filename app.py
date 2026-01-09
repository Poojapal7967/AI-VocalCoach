import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import os
import time

# --- 1. ULTRA-NEON ATTRATIVE UI (Pixel Perfect) ---
st.set_page_config(page_title="AI Vocal Coach Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #08081a; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* Recording Pulse Animation (image_302c23 style) */
    @keyframes pulse-red {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 12px rgba(255, 75, 75, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    .recording-dot {
        width: 14px; height: 14px; background: #ff4b4b; border-radius: 50%;
        display: inline-block; margin-right: 10px; animation: pulse-red 1.5s infinite;
    }

    /* Glassmorphism Cards with Precise Glow (image_31767c) */
    .neon-card {
        background: rgba(255, 255, 255, 0.04); border-radius: 25px; padding: 45px 20px;
        text-align: center; backdrop-filter: blur(20px); height: 320px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.1); transition: 0.5s;
    }
    .glow-red { border-color: #ff4b4b !important; box-shadow: 0 0 30px rgba(255, 75, 75, 0.5); }
    .glow-orange { border-color: #ff9800 !important; box-shadow: 0 0 30px rgba(255, 152, 0, 0.5); }
    .glow-cyan { border-color: #00f2fe !important; box-shadow: 0 0 30px rgba(0, 242, 254, 0.5); }

    .big-num { font-size: 85px; font-weight: 900; line-height: 1; margin-bottom: 10px; }
    .label-sub { font-size: 14px; color: #a0a0b0; text-transform: uppercase; letter-spacing: 3px; }

    /* Audio Player Widget UI Fix */
    div.stAudio { margin-top: 15px; width: 95% !important; border-radius: 12px; }

    /* Neon Buttons (image_317e3d) */
    .stButton > button {
        width: 100% !important; border-radius: 50px !important;
        padding: 15px 0px !important; font-weight: 800 !important;
        text-transform: uppercase; letter-spacing: 2px; transition: 0.3s;
    }
    /* Start Button Red Glow */
    div[data-testid="column"]:nth-of-type(1) .stButton > button {
        border: 2.5px solid #ff4b4b !important; box-shadow: 0 0 25px rgba(255, 75, 75, 0.6);
        background: transparent !important; color: white !important;
    }
    /* Analyze Button White Glow */
    div[data-testid="column"]:nth-of-type(2) .stButton > button {
        border: 2.5px solid #ffffff !important; box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
        background: transparent !important; color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC INITIALIZATION ---
if 'recording' not in st.session_state: st.session_state.recording = False
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False

@st.cache_resource
def load_model(): return whisper.load_model("base")
model = load_model()

# --- 3. PERFORMANCE GRID ---
st.write("##")
st.markdown('<p style="color:#ffcc00; font-weight:800; font-size:20px; letter-spacing:1px;">📊 PERFORMANCE REPORT</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="neon-card glow-red"><div class="big-num">19</div><div class="label-sub">Confidence Score</div></div>', unsafe_allow_html=True)

with c2:
    # Arranged Audio Player Inside Card (image_302c23 style)
    st.markdown('<div class="neon-card glow-orange">', unsafe_allow_html=True)
    if st.session_state.recording:
        st.markdown('<div><span class="recording-dot"></span><span style="color:#ff4b4b; font-weight:bold;">LIVE RECORDING</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:45px; margin-bottom:10px;">🎙️</div>', unsafe_allow_html=True)
    
    if os.path.exists("speech.wav"):
        st.audio("speech.wav")
        st.markdown('<div class="label-sub">SPEECH.WAV RECORDED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="label-sub">NO DATA FOUND</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="neon-card glow-cyan"><div class="big-num">3%</div><div class="label-sub">Fillers Found</div></div>', unsafe_allow_html=True)

# --- 4. START & ANALYZE CONTROLS ---
st.write("##")
_, c_btn, _ = st.columns([1, 2, 1])

with c_btn:
    b1, b2 = st.columns(2)
    with b1:
        if st.button("🎤 START"):
            st.session_state.recording = True
            st.session_state.analysis_ready = False
            fs = 44100
            duration = 5 
            st.toast("Recording for 5 seconds...", icon="🔴")
            rec_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            write("speech.wav", fs, rec_data)
            st.session_state.recording = False
            st.session_state.analysis_ready = True
            st.rerun()

    with b2:
        if st.button("🛑 ANALYZE"):
            if os.path.exists("speech.wav"):
                st.toast("Whisper AI is processing...", icon="⌛")
                result = model.transcribe("speech.wav")
                st.session_state.transcription = result['text']
                st.rerun()
            else:
                st.warning("Please record your voice first!")

# --- 5. COACHING FEEDBACK ---
if st.session_state.analysis_ready and 'transcription' in st.session_state:
    st.write("##")
    st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); border: 1.5px solid #00f2fe; border-radius: 20px; padding: 35px; box-shadow: 0 0 20px rgba(0, 242, 254, 0.2);">
            <p style="color:#00f2fe; font-weight:900; font-size:20px; text-transform:uppercase; letter-spacing:1px;">💡 Coaching Feedback</p>
            <p style="color:#fff; font-size:18px; font-style:italic; margin-bottom:20px;">"{st.session_state.transcription}"</p>
            <hr style="border-color:rgba(255,255,255,0.1);">
            <div style="color:#4caf50; font-weight:600;">★ Success: Voice captured and transcribed successfully!</div>
            <div style="color:#8e949a; font-size:14px; margin-top:10px;">Tips: Focus on clarity and maintain this stable pitch.</div>
        </div>
    """, unsafe_allow_html=True)