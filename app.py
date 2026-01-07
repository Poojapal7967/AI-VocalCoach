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

# --- PREMIUM VOCAL IMAGE UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    /* Dark Theme & Typography */
    .stApp { background: #0c0c1e; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* Hero Title with Gradient */
    .hero-title { 
        font-size: 55px; font-weight: 800; 
        background: linear-gradient(90deg, #ffffff, #7000ff); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
        margin-bottom: 0px; text-align: center;
    }
    
    /* Feature Cards (image_96fbc6.jpg inspired) */
    .feature-card { 
        background: rgba(255, 255, 255, 0.05); 
        border-radius: 20px; padding: 25px; 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        text-align: left; height: 180px;
    }
    
    /* Yellow Action Buttons (image_96fb29.png inspired) */
    div.stButton > button:first-child {
        background: #ffcc00 !important; color: #000 !important;
        font-weight: 800 !important; border-radius: 15px !important;
        border: none !important; padding: 15px 30px !important;
        font-size: 18px !important; transition: 0.3s;
    }
    div.stButton > button:hover { transform: scale(1.05); box-shadow: 0px 0px 20px rgba(255, 204, 0, 0.4); }

    /* Centering Content */
    .main .block-container { display: flex; flex-direction: column; align-items: center; }
    
    /* Table Styling */
    .stTable { background: rgba(255,255,255,0.05); border-radius: 15px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

# --- DATABASE LOGIC ---
HISTORY_FILE = 'vocal_history.json'

def save_to_history(new_data):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f: history = json.load(f)
    history.append(new_data)
    with open(HISTORY_FILE, 'w') as f: json.dump(history, f, indent=4)

# --- SESSION STATES ---
if 'recording_start' not in st.session_state: st.session_state.recording_start = None
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False

@st.cache_resource
def load_model(): return whisper.load_model("base")
model = load_model()

# --- 1. HERO SECTION ---
st.markdown('<p class="hero-title">Transform Your Voice</p>', unsafe_allow_html=True)
st.write("<p style='text-align:center; font-size:20px; color:#aaa;'>AI-Powered Coaching for Professional Communication</p>", unsafe_allow_html=True)

st.write("##")

# --- 2. FEATURE CARDS ---
c_f1, c_f2, c_f3 = st.columns(3)
with c_f1: st.markdown('<div class="feature-card"><h3 style="color:#7000ff;">🎭 Voice Rating</h3><p>Get a professional score on your pitch and energy levels.</p></div>', unsafe_allow_html=True)
with c_f2: st.markdown('<div class="feature-card"><h3 style="color:#00f2fe;">🌍 Accent Check</h3><p>Analyze your clarity and pronunciation accuracy.</p></div>', unsafe_allow_html=True)
with c_f3: st.markdown('<div class="feature-card"><h3 style="color:#ffcc00;">📈 Progress</h3><p>Track your speech journey with stored session history.</p></div>', unsafe_allow_html=True)

st.write("##")
st.markdown("---")

# --- 3. ANALYZER SECTION ---
st.subheader("Think you speak clearly? Prove it.")

col_set1, col_set2 = st.columns(2)
with col_set1: language = st.selectbox("Language", ["English", "Hindi"])
with col_set2: goal = st.selectbox("Goal", ["Public Speaking", "Anchoring", "Teaching", "Interview"])

# Control Buttons
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

if b3.button("🔄 RESET UI"):
    st.session_state.analysis_ready = False
    st.rerun()

if b4.button("🗑️ CLEAR DATA"):
    if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
    st.success("History Cleared!")
    st.rerun()

if st.session_state.recording_start:
    st.warning("🔴 Recording in Progress... Speak into the mic.")

# --- 4. RESULTS ---
if st.session_state.analysis_ready and os.path.exists('speech.wav'):
    st.audio('speech.wav')
    
    with st.spinner("AI is calculating your score..."):
        y, sr = librosa.load('speech.wav')
        result = model.transcribe('speech.wav', language=("en" if language=="English" else "hi"))
        text = result['text']
        
        actual_time = len(y) / sr
        wpm = int(len(text.split()) / (actual_time / 60)) if actual_time > 0 else 0
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        # Save to Storage
        session_data = {
            "Date": datetime.now().strftime("%d %b, %H:%M"),
            "Goal": goal,
            "Pace": f"{wpm} WPM",
            "Energy": "High" if pitch_var > 60 else "Low"
        }
        save_to_history(session_data)

        st.markdown(f"### 📊 Performance Report")
        
        # Purple Waveform
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#7000ff')
        ax.set_axis_off()
        fig.patch.set_facecolor('#0c0c1e')
        st.pyplot(fig)

        m1, m2, m3 = st.columns(3)
        m1.metric("Speech Pace", session_data["Pace"])
        m2.metric("Duration", f"{round(actual_time, 1)}s")
        m3.metric("Voice Energy", session_data["Energy"])

        st.info(f"**Transcription:** {text}")

# --- 5. HISTORY TABLE ---
st.write("##")
st.subheader("📜 Your Speech Journey")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'r') as f:
        data = json.load(f)
        st.table(data[::-1])
else:
    st.write("No sessions recorded yet. Start your first test now!")