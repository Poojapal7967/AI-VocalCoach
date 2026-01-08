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
import json
from datetime import datetime
import pandas as pd
import plotly.express as px

# --- 1. EXCLUSIVE NEON GLASS UI (Inspired by image_302c23.jpg) ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #0c0c1e; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* Hero Title with Neon Glow */
    .hero-title { 
        font-size: 60px; font-weight: 800; 
        background: linear-gradient(90deg, #ffffff, #ffcc00, #7000ff); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
        text-align: center; margin-bottom: 0px;
        filter: drop-shadow(0px 0px 10px rgba(112, 0, 255, 0.3));
    }
    
    /* Glassmorphic Report Box (image_302c23.jpg) */
    .report-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        padding: 40px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }
    
    /* Neon Coaching Tips Box */
    .tips-box {
        background: rgba(0, 242, 254, 0.05);
        border-left: 5px solid #00f2fe;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }

    /* Buttons with Glow */
    div.stButton > button {
        background: linear-gradient(135deg, #ffcc00 0%, #ffb300 100%) !important;
        color: #000 !important; font-weight: 800 !important;
        border-radius: 20px !important; border: none !important;
        padding: 15px 35px !important; font-size: 18px !important;
        box-shadow: 0px 4px 15px rgba(255, 204, 0, 0.3);
    }
    
    .filler-err { color: #ff4b4b; font-weight: bold; border-bottom: 2px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC FUNCTIONS ---
HISTORY_FILE = 'vocal_history.json'
def save_to_history(new_data):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f: history = json.load(f)
    history.append(new_data)
    with open(HISTORY_FILE, 'w') as f: json.dump(history, f, indent=4)

if 'active_test' not in st.session_state: st.session_state.active_test = "Voice Rating"
if 'recording_start' not in st.session_state: st.session_state.recording_start = None
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False

@st.cache_resource
def load_model(): return whisper.load_model("base")
model = load_model()

# --- 3. DYNAMIC NAVIGATION (image_97f6a9.png Style) ---
st.markdown('<p class="hero-title">Transform Your Voice</p>', unsafe_allow_html=True)
st.write("<p style='text-align:center; color:#888; font-size:18px;'>AI-Powered Feedback for Professional Communication</p>", unsafe_allow_html=True)

st.write("##")
c_f1, c_f2, c_f3 = st.columns(3)

with c_f1:
    if st.button("🎭 Voice Rating"): st.session_state.active_test = "Voice Rating"
    st.markdown(f'<div class="feature-card {"active-card" if st.session_state.active_test=="Voice Rating" else ""}"><h3>Rating</h3><p>Score pitch & energy.</p></div>', unsafe_allow_html=True)
with c_f2:
    if st.button("🌍 Clarity Check"): st.session_state.active_test = "Clarity Check"
    st.markdown(f'<div class="feature-card {"active-card" if st.session_state.active_test=="Clarity Check" else ""}"><h3>Clarity</h3><p>Detect filler words.</p></div>', unsafe_allow_html=True)
with c_f3:
    if st.button("📈 Progress"): st.session_state.active_test = "Progress"
    st.markdown(f'<div class="feature-card {"active-card" if st.session_state.active_test=="Progress" else ""}"><h3>History</h3><p>View growth trends.</p></div>', unsafe_allow_html=True)

st.write("---")

# --- 4. ANALYZER INTERFACE (image_302c23.jpg Style) ---
if st.session_state.active_test == "Progress":
    st.subheader("📈 Performance Trends")
    if os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(json.load(open(HISTORY_FILE)))
        df['WPM_Num'] = df['Pace'].str.extract('(\d+)').astype(int)
        fig = px.line(df, x='Date', y='WPM_Num', title='Consistency Trend', template='plotly_dark')
        fig.update_traces(line_color='#ffcc00', mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No data yet!")

else:
    col_l, col_r = st.columns([1, 1.5])
    
    with col_l:
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.write("### ⚙️ Settings")
        language = st.selectbox("Language", ["English", "Hindi"])
        goal = st.selectbox("Goal", ["Public Speaking", "Interview", "Teaching"])
        
        st.write("##")
        if st.button("🎤 START RECORDING", type="primary"):
            st.session_state.recording_start = time.time()
            st.session_state.raw_audio = sd.rec(int(300 * 44100), samplerate=44100, channels=1)
            st.rerun()
            
        if st.button("🛑 STOP & ANALYZE"):
            if st.session_state.recording_start:
                sd.stop()
                st.session_state.analysis_ready = True
                st.session_state.recording_start = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        if st.session_state.analysis_ready:
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.write("### 📊 Performance Report")
            # Analysis Logic ...
            y, sr = librosa.load('speech.wav')
            result = model.transcribe('speech.wav')
            text = result['text']
            
            words = text.lower().split()
            fillers = ["um", "uh", "ah", "matlab", "toh"]
            found_fillers = [w for w in words if w in fillers]
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", "85%")
            m2.metric("Pace", f"{int(len(words)/(len(y)/sr/60))} WPM")
            m3.metric("Fillers", f"{len(found_fillers)}")
            
            st.markdown("---")
            st.write("### 💡 Coaching Tips")
            st.markdown("""
            * 🌟 **Success:** Your tone is professional.
            * ⚠️ **Warning:** Slow down during complex sentences.
            * 🚀 **Pro Tip:** Pause for 2 seconds after each key point.
            """)
            st.markdown('</div>', unsafe_allow_html=True)