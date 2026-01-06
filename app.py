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

# --- ULTIMATE NEON GLASS UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at center, #1a1a3a 0%, #050510 100%); color: #ffffff; }
    .main .block-container { text-align: center; display: flex; flex-direction: column; align-items: center; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 50px !important; font-weight: 900 !important; display: flex; justify-content: center; }
    [data-testid="stMetricLabel"] { color: #00f2fe !important; font-size: 18px !important; display: flex; justify-content: center; width: 100%; }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.08) !important; backdrop-filter: blur(20px) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 25px !important; padding: 30px !important; text-align: center; }
    .improvement-box { background: rgba(255, 75, 75, 0.1); border: 1px solid #ff4b4b; padding: 20px; border-radius: 15px; margin: 10px auto; max-width: 800px; text-align: center; }
    .stButton>button { width: 100% !important; border-radius: 50px !important; padding: 15px !important; font-size: 18px !important; font-weight: bold !important; border: none !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- HISTORY DATABASE FUNCTIONS ---
HISTORY_FILE = 'vocal_history.json'

def save_to_history(new_data):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    history.append(new_data)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def delete_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    if os.path.exists('speech.wav'):
        os.remove('speech.wav')

# --- SESSION STATES ---
if 'recording_start' not in st.session_state: st.session_state.recording_start = None
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False

@st.cache_resource
def load_model(): return whisper.load_model("base")
model = load_model()

# --- APP UI ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("Master Your Voice & Track Your Progress History")
st.markdown("---")

c1, c2 = st.columns(2)
with c1: language = st.selectbox("Language", ["English", "Hindi"])
with c2: goal = st.selectbox("Goal", ["Public Speaking", "Anchoring", "Teaching", "Interview"])

st.write("##")

# --- CONTROL BUTTONS (4 Columns) ---
b1, b2, b3, b4 = st.columns(4)

if b1.button("🎤 START", type="primary"):
    st.session_state.recording_start = time.time()
    st.session_state.analysis_ready = False
    st.session_state.raw_audio = sd.rec(int(300 * 44100), samplerate=44100, channels=1)
    st.rerun()

if b2.button("🛑 STOP"):
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

if b4.button("🗑️ DELETE HISTORY"):
    delete_history()
    st.session_state.analysis_ready = False
    st.success("All History Deleted!")
    time.sleep(1)
    st.rerun()

if st.session_state.recording_start:
    st.warning("🔴 Recording in Progress...")

# --- ANALYSIS & HISTORY SECTION ---
if st.session_state.analysis_ready and os.path.exists('speech.wav'):
    st.audio('speech.wav')
    
    with st.spinner("⌛ AI Analyzing..."):
        y, sr = librosa.load('speech.wav')
        result = model.transcribe('speech.wav', language=("en" if language=="English" else "hi"))
        text = result['text']
        
        actual_time = len(y) / sr
        wpm = int(len(text.split()) / (actual_time / 60)) if actual_time > 0 else 0
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        # Save to JSON database
        session_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "goal": goal,
            "pace": f"{wpm} WPM",
            "time": f"{round(actual_time, 1)}s",
            "energy": "High" if pitch_var > 60 else "Low"
        }
        save_to_history(session_data)

        st.markdown(f"### 📊 Current Session Analysis: {goal}")
        
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#7000ff')
        ax.set_axis_off()
        fig.patch.set_facecolor('#050510')
        st.pyplot(fig)

        m1, m2, m3 = st.columns(3)
        m1.metric("Pace", session_data["pace"])
        m2.metric("Time", session_data["time"])
        m3.metric("Energy", session_data["energy"])

        st.info(f"**Transcription:** {text}")

# --- HISTORY TABLE ---
st.markdown("---")
st.subheader("📜 Session History (All Stored Data)")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'r') as f:
        data = json.load(f)
        # Latest recordings ko sabse upar dikhane ke liye reverse kiya
        st.table(data[::-1])
else:
    st.write("No history found. Start your first session!")