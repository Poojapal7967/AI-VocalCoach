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

# --- 1. PREMIUM VOCAL IMAGE UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    /* Dark Theme & Typography */
    .stApp { background: #0c0c1e; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* Hero Title with Gradient */
    .hero-title { 
        font-size: 55px; font-weight: 800; 
        background: linear-gradient(90deg, #ffffff, #ffcc00, #7000ff); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
        margin-bottom: 0px; text-align: center;
    }
    
    /* Feature Cards */
    .feature-card { 
        background: rgba(255, 255, 255, 0.05); 
        border-radius: 20px; padding: 25px; 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        text-align: left; height: 180px; transition: 0.3s;
    }
    .feature-card:hover { border: 1px solid #7000ff; transform: translateY(-5px); }
    
    /* Yellow Action Buttons */
    div.stButton > button {
        background: #ffcc00 !important; color: #000 !important;
        font-weight: 800 !important; border-radius: 15px !important;
        border: none !important; padding: 15px 30px !important;
        font-size: 18px !important; width: 100%;
    }
    div.stButton > button:hover { box-shadow: 0px 0px 20px rgba(255, 204, 0, 0.4); }

    /* Centering content and styling metrics */
    .main .block-container { display: flex; flex-direction: column; align-items: center; }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 15px; border: 1px solid rgba(255, 255, 255, 0.1); }
    
    /* Error Highlighting */
    .filler-err { color: #ff4b4b; font-weight: bold; border-bottom: 2px solid #ff4b4b; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE LOGIC ---
HISTORY_FILE = 'vocal_history.json'

def save_to_history(new_data):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f: history = json.load(f)
    history.append(new_data)
    with open(HISTORY_FILE, 'w') as f: json.dump(history, f, indent=4)

# --- 3. SESSION STATES ---
if 'recording_start' not in st.session_state: st.session_state.recording_start = None
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False

@st.cache_resource
def load_model(): return whisper.load_model("base")
model = load_model()

# --- 4. HERO SECTION ---
st.markdown('<p class="hero-title">Transform Your Voice</p>', unsafe_allow_html=True)
st.write("<p style='text-align:center; font-size:20px; color:#aaa;'>AI-Powered Coaching for Professional Communication</p>", unsafe_allow_html=True)

st.write("##")

# Feature Cards (image_96fbc6.jpg Style)
c_f1, c_f2, c_f3 = st.columns(3)
with c_f1: st.markdown('<div class="feature-card"><h3 style="color:#7000ff;">🎭 Voice Rating</h3><p>Get a professional score on your pitch and energy levels.</p></div>', unsafe_allow_html=True)
with c_f2: st.markdown('<div class="feature-card"><h3 style="color:#00f2fe;">🌍 Clarity Check</h3><p>Analyze your pronunciation accuracy and filler word frequency.</p></div>', unsafe_allow_html=True)
with c_f3: st.markdown('<div class="feature-card"><h3 style="color:#ffcc00;">📈 Progress</h3><p>Track your speech journey with stored session history and graphs.</p></div>', unsafe_allow_html=True)

st.write("##")
st.markdown("---")

# --- 5. ANALYZER SECTION ---
st.subheader("Think you speak clearly? Prove it.")

col_set1, col_set2 = st.columns(2)
with col_set1: language = st.selectbox("Language", ["English", "Hindi"])
with col_set2: goal = st.selectbox("Goal", ["Public Speaking", "Anchoring", "Teaching", "Interview"])

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
    st.warning("🔴 Recording in Progress... Speak clearly into the mic.")

# --- 6. RESULTS & ERROR DETECTION ---
if st.session_state.analysis_ready and os.path.exists('speech.wav'):
    st.audio('speech.wav')
    
    with st.spinner("AI is scanning for vocal errors..."):
        y, sr = librosa.load('speech.wav')
        result = model.transcribe('speech.wav')
        text = result['text']
        
        # Error Detection Logic (Filler words)
        fillers = ["um", "uh", "ah", "like", "hmm", "matlab", "toh", "basically", "actually"]
        words = text.lower().split()
        found_fillers = [w for w in words if w.strip(",.") in fillers]
        
        actual_time = len(y) / sr
        wpm = int(len(words) / (actual_time / 60)) if actual_time > 0 else 0
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        # Save Entry
        save_to_history({
            "Date": datetime.now().strftime("%d %b, %H:%M"),
            "Goal": goal, "Pace": f"{wpm} WPM", "Energy": "High" if pitch_var > 60 else "Low", "Fillers": len(found_fillers)
        })

        st.markdown(f"### 📊 Performance Report")
        
        # Purple Waveform
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#7000ff')
        ax.set_axis_off()
        fig.patch.set_facecolor('#0c0c1e')
        st.pyplot(fig)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Speech Pace", f"{wpm} WPM")
        m2.metric("Duration", f"{round(actual_time, 1)}s")
        m3.metric("Voice Energy", "High" if pitch_var > 60 else "Low")
        m4.metric("Filler Words", len(found_fillers))

        # Red Highlighting for Transcription Errors (image_85a089.jpg Style)
        st.write("##")
        st.subheader("📝 Vocal Error Feedback (Errors in Red)")
        display_html = ""
        for word in words:
            if word.strip(",.") in fillers:
                display_html += f'<span class="filler-err">{word}</span> '
            else:
                display_html += f"{word} "
        st.markdown(f'<div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; border:1px solid rgba(255,255,255,0.1);">{display_html}</div>', unsafe_allow_html=True)

# --- 7. HISTORY & TRENDS ---
st.write("---")
if os.path.exists(HISTORY_FILE):
    st.subheader("📜 Your Speech Journey")
    with open(HISTORY_FILE, 'r') as f:
        data = json.load(f)
        df = pd.DataFrame(data)
        if not df.empty:
            # Pace Trend Graph (image_9760f9.png Style)
            df['WPM_Num'] = df['Pace'].str.extract('(\d+)').astype(int)
            fig_trend = px.line(df, x='Date', y='WPM_Num', title='Pace Consistency Trend', template='plotly_dark')
            fig_trend.update_traces(line_color='#ffcc00', mode='lines+markers')
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # History Table
            st.table(df[::-1])
else:
    st.write("No sessions recorded yet. Start your first test!")

    if os.path.exists(HISTORY_FILE):
     st.subheader("📜 Your Speech Journey")
    with open(HISTORY_FILE, 'r') as f:
        data = json.load(f)
        df = pd.DataFrame(data)
        if not df.empty:
            # WPM column se number nikalne ke liye
            df['WPM_Num'] = df['Pace'].str.extract('(\d+)').astype(int)
            
            # Graph create karein
            fig_trend = px.line(df, x='Date', y='WPM_Num', title='Pace Consistency Trend', template='plotly_dark')
            
            # FIX: mode='lines+markers' property sahi hai
            fig_trend.update_traces(line_color='#ffcc00', mode='lines+markers')
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # History Table ko graph ke niche dikhayein
            st.table(df[::-1])