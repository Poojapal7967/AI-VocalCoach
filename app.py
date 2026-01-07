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

# --- 1. PREMIUM UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #0c0c1e; color: #ffffff; font-family: 'Inter', sans-serif; }
    .hero-title { 
        font-size: 55px; font-weight: 800; 
        background: linear-gradient(90deg, #ffffff, #ffcc00, #7000ff); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
        text-align: center; margin-bottom: 5px;
    }
    .feature-card { 
        background: rgba(255, 255, 255, 0.05); 
        border-radius: 20px; padding: 25px; 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        height: 180px; transition: 0.3s;
    }
    .active-card { border: 2px solid #7000ff !important; background: rgba(112, 0, 255, 0.1) !important; }
    
    div.stButton > button {
        background: #ffcc00 !important; color: #000 !important;
        font-weight: 800 !important; border-radius: 15px !important;
        font-size: 16px !important; width: 100%;
    }
    div.stButton > button:hover { transform: scale(1.02); }
    .filler-err { color: #ff4b4b; font-weight: bold; border-bottom: 2px solid #ff4b4b; }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE LOGIC ---
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

# --- 3. HERO & CLICKABLE CARDS ---
st.markdown('<p class="hero-title">Transform Your Voice</p>', unsafe_allow_html=True)
st.write("<p style='text-align:center; color:#aaa; font-size:18px;'>AI-Powered Feedback for Professional Communication</p>", unsafe_allow_html=True)

st.write("##")
c_f1, c_f2, c_f3 = st.columns(3)

with c_f1:
    if st.button("🎭 Select Voice Rating"): st.session_state.active_test = "Voice Rating"
    is_active = "active-card" if st.session_state.active_test == "Voice Rating" else ""
    st.markdown(f'<div class="feature-card {is_active}"><h3>Voice Rating</h3><p>Score your pitch and energy levels.</p></div>', unsafe_allow_html=True)

with c_f2:
    if st.button("🌍 Select Clarity Check"): st.session_state.active_test = "Clarity Check"
    is_active = "active-card" if st.session_state.active_test == "Clarity Check" else ""
    st.markdown(f'<div class="feature-card {is_active}"><h3>Clarity Check</h3><p>Detect fillers and accent accuracy.</p></div>', unsafe_allow_html=True)

with c_f3:
    if st.button("📈 Select Progress"): st.session_state.active_test = "Progress"
    is_active = "active-card" if st.session_state.active_test == "Progress" else ""
    st.markdown(f'<div class="feature-card {is_active}"><h3>Progress</h3><p>Track history and view growth graphs.</p></div>', unsafe_allow_html=True)

st.write("---")

# --- 4. DYNAMIC SECTION SWITCHER ---
if st.session_state.active_test == "Progress":
    st.subheader("📊 Your Speech Journey & Trends")
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            data_list = json.load(f)
            df = pd.DataFrame(data_list)
            if not df.empty:
                df['WPM_Num'] = df['Pace'].str.extract('(\d+)').astype(int)
                fig = px.line(df, x='Date', y='WPM_Num', title='Pace Consistency', template='plotly_dark')
                fig.update_traces(line_color='#ffcc00', mode='lines+markers')
                st.plotly_chart(fig, use_container_width=True)
                st.table(df[::-1])
    else:
        st.info("Start your first test to see progress!")

else:
    st.subheader(f"Current Mode: {st.session_state.active_test}")
    col_set1, col_set2 = st.columns(2)
    with col_set1: language = st.selectbox("Language", ["English", "Hindi"])
    with col_set2: goal = st.selectbox("Goal", ["Public Speaking", "Interview", "Teaching"])

    st.write("##")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("🎤 START TEST"):
        st.session_state.recording_start = time.time()
        st.session_state.analysis_ready = False
        st.session_state.raw_audio = sd.rec(int(300 * 44100), samplerate=44100, channels=1)
        st.rerun()

    if b2.button("🛑 STOP & ANALYZE"):
        if st.session_state.recording_start:
            duration = time.time() - st.session_state.recording_start
            sd.stop()
            cropped = st.session_state.raw_audio[:int(duration * 44100)]
            if np.max(np.abs(cropped)) > 0: cropped = cropped / np.max(np.abs(cropped))
            write('speech.wav', 44100, cropped)
            st.session_state.analysis_ready = True
            st.session_state.recording_start = None
            st.rerun()
    
    if b3.button("🔄 RESET"):
        st.session_state.analysis_ready = False
        st.rerun()
    
    if b4.button("🗑️ CLEAR"):
        if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
        st.rerun()

    if st.session_state.recording_start: st.warning("🔴 Recording in Progress...")

    # --- ANALYSIS RESULTS ---
    if st.session_state.analysis_ready and os.path.exists('speech.wav'):
        st.audio('speech.wav')
        with st.spinner("AI is calculating your score..."):
            y, sr = librosa.load('speech.wav')
            result = model.transcribe('speech.wav')
            text = result['text']
            
            fillers = ["um", "uh", "ah", "like", "matlab", "toh", "basically", "actually"]
            words = text.lower().split()
            found_fillers = [w for w in words if w.strip(",.") in fillers]
            
            wpm = int(len(words) / ((len(y)/sr) / 60)) if len(y) > 0 else 0
            pitches, _ = librosa.piptrack(y=y, sr=sr)
            pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

            # Dashboard Display
            st.markdown(f"### 📊 Analysis Report")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Speech Pace", f"{wpm} WPM")
            m2.metric("Duration", f"{round(len(y)/sr, 1)}s")
            m3.metric("Voice Energy", "High" if pitch_var > 60 else "Low")
            m4.metric("Filler Words", len(found_fillers))

            # Waveform
            fig, ax = plt.subplots(figsize=(10, 2))
            librosa.display.waveshow(y, sr=sr, ax=ax, color='#7000ff')
            ax.set_axis_off()
            fig.patch.set_facecolor('#0c0c1e')
            st.pyplot(fig)

            # Highlighting
            st.subheader("📝 Transcription Feedback (Errors in Red)")
            display_html = "".join([f'<span class="filler-err">{w}</span> ' if w.strip(",.") in fillers else f"{w} " for w in words])
            st.markdown(f'<div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; border:1px solid rgba(255,255,255,0.1);">{display_html}</div>', unsafe_allow_html=True)

            save_to_history({"Date": datetime.now().strftime("%d %b, %H:%M"), "Goal": goal, "Pace": f"{wpm} WPM", "Fillers": len(found_fillers)})