import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import os
import time
import json
from datetime import datetime
import pandas as pd
import plotly.express as px

# --- 1. GLASSMORPISM & NEON UI STYLING (Inspired by image_302c23.jpg) ---
st.set_page_config(page_title="AI Vocal Coach Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #0c0c1e; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* Hero Section */
    .hero-title { 
        font-size: 50px; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #ffffff, #ffcc00, #7000ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* Glassmorphism Performance Box (image_302c23.jpg) */
    .performance-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        padding: 30px;
        backdrop-filter: blur(15px);
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Start Recording Neon Button */
    div.stButton > button {
        background: transparent !important;
        color: #ffffff !important;
        border: 2px solid #ff4b4b !important;
        border-radius: 30px !important;
        padding: 15px 30px !important;
        font-weight: bold !important;
        box-shadow: 0px 0px 15px rgba(255, 75, 75, 0.4);
        transition: 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #ff4b4b !important;
        box-shadow: 0px 0px 25px rgba(255, 75, 75, 0.6);
    }

    /* Coaching Tips Styling */
    .tips-header { 
        background: rgba(0, 242, 254, 0.1); 
        border: 1px solid #00f2fe; 
        padding: 15px; border-radius: 12px;
        margin-top: 15px;
    }
    
    .tip-item { font-size: 16px; margin-bottom: 8px; display: flex; align-items: center; }
    .tip-icon { margin-right: 10px; }

    /* Metric Styling */
    [data-testid="stMetricValue"] { font-size: 45px !important; font-weight: 800 !important; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC FUNCTIONS ---
HISTORY_FILE = 'vocal_history.json'

if 'active_test' not in st.session_state: st.session_state.active_test = "Voice Rating"
if 'recording_start' not in st.session_state: st.session_state.recording_start = None
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False

@st.cache_resource
def load_model(): return whisper.load_model("base")
model = load_model()

# --- 3. MAIN INTERFACE ---
st.markdown('<p class="hero-title">AI Vocal Coach Pro</p>', unsafe_allow_html=True)

# Navigation Cards (Like image_97f6a9.png)
col_nav1, col_nav2, col_nav3 = st.columns(3)
with col_nav1:
    if st.button("🎭 Voice Rating"): st.session_state.active_test = "Voice Rating"
with col_nav2:
    if st.button("🌍 Clarity Check"): st.session_state.active_test = "Clarity Check"
with col_nav3:
    if st.button("📈 Progress Graph"): st.session_state.active_test = "Progress"

st.write("---")

if st.session_state.active_test == "Progress":
    # (Aapka plotly graph logic yahan aayega)
    st.subheader("📈 Performance History")
    if os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(json.load(open(HISTORY_FILE)))
        fig = px.line(df, x='Date', y='Pace', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No data yet.")

else:
    # --- PERFORMANCE REPORT UI (As per image_302c23.jpg) ---
    st.markdown('<div class="performance-card">', unsafe_allow_html=True)
    st.write("### 📊 Performance Report")
    
    met1, met2, met3 = st.columns([1, 1.2, 1])
    
    with met1:
        st.metric("Confidence Score", "19") # Sample Data
        if st.button("🔴 Start Recording"):
            st.session_state.recording_start = time.time()
            st.session_state.raw_audio = sd.rec(int(300 * 44100), samplerate=44100, channels=1)
            st.rerun()

    with met2:
        # Central Visualizer Placeholder
        st.write("##")
        st.info("🔊 speech.wav")
        st.caption("transcribe.wav ready")

    with met3:
        st.metric("Fillers Found", "3%")
        st.caption("Speaking Speed: Normal")

    st.markdown('</div>', unsafe_allow_html=True)

    # --- COACHING TIPS (As per image_302c23.jpg) ---
    st.markdown('<div class="tips-header">', unsafe_allow_html=True)
    st.write("💡 **Coaching Tips**")
    st.markdown("""
    <div class="tip-item"><span class="tip-icon">🟢</span> Success: Your tone is very stable today.</div>
    <div class="tip-item"><span class="tip-icon">🟠</span> Warning: Slow down during the first 10 seconds.</div>
    <div class="tip-item"><span class="tip-icon">🔴</span> Filler: "Um" and "Ah" detected more than usual.</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)