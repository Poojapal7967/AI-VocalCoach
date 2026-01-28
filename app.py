import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import pandas as pd 
import google.generativeai as genai 

# --- 0. AI CONFIG ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-pro')
except:
    st.error("Invalid API Key. Update Gemini Key in code.")

# --- 1. STATE MANAGEMENT ---
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'vocal_history' not in st.session_state: st.session_state.vocal_history = [] 

# --- 2. OPTIMIZED MODEL LOADING (FIX FOR ANALYSIS) ---
@st.cache_resource
def load_whisper_model():
    # Model ko cache karne se system memory crash nahi hoti
    return whisper.load_model("base")

# --- 3. ELITE DARK UI STYLING ---
st.set_page_config(page_title="Speeko Elite", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #05070A; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    .hero-box {
        background: radial-gradient(circle at 70% 30%, rgba(108, 92, 231, 0.15) 0%, rgba(5, 7, 10, 1) 70%);
        padding: 60px 40px; border-radius: 30px; margin-bottom: 30px; text-align: center;
    }
    .stButton>button { 
        background: linear-gradient(90deg, #FF00CC, #3333FF); color: white; border-radius: 12px; 
        padding: 12px 30px; border: none; font-weight: 800; transition: 0.3s; width: 100%;
    }
    .delete-container button { background: linear-gradient(90deg, #FF4B4B, #FF0000) !important; }
    .metric-card { background: #0D1117; border-radius: 20px; padding: 25px; border: 1px solid #30363D; margin-bottom: 20px; }
    .glass-card {
        background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px); border-radius: 24px; padding: 40px 20px; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. CORE ANALYTICS ENGINE ---
def get_full_analysis(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    variation = float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    energy_label = "High Impact" if variation >= 18 else "Moderate"
    intervals = librosa.effects.split(y, top_db=25)
    speech_time = sum([s[1] - s[0] for s in intervals]) / sr
    talk_ratio = (speech_time / (len(y)/sr)) * 100 if len(y) > 0 else 0
    return int(tempo[0] if isinstance(tempo, np.ndarray) else tempo), energy_label, int(talk_ratio)

# --- 5. NAVIGATION ENGINE (ROBUST) ---
if st.session_state.page == 'home':
    st.markdown('<div class="hero-box"><h1 style="font-size:4rem; font-weight:900;">Speak <span style="color: #6C5CE7;"><</span>Confidently.</h1><p>Master Every Mode of Communication</p></div>', unsafe_allow_html=True)
    if st.button("🚀 ENTER DASHBOARD"):
        st.session_state.page = 'dashboard'
        st.rerun()
    f1, f2, f3 = st.columns(3)
    features = [("📊", "Metrics", "BPM analysis."), ("🧠", "AI Bot", "Practice scripts."), ("📈", "Progress", "Track growth charts.")]
    for col, (i, t, d) in zip([f1, f2, f3], features):
        col.markdown(f'<div class="glass-card"><h1>{i}</h1><h3>{t}</h3><p>{d}</p></div>', unsafe_allow_html=True)

else:
    with st.sidebar:
        st.markdown("""<div style="background:linear-gradient(90deg, #FF00CC, #3333FF); padding:10px; border-radius:10px; text-align:center;"><h3 style="margin:0;">Speeko Elite</h3></div>""", unsafe_allow_html=True)
        if st.button("⬅️ HOME"):
            st.session_state.page = 'home'
            st.rerun()
        st.divider()
        nav = st.radio("Menu", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Script Bot"])
        rec_dur = st.slider("Duration (s):", 5, 30, 10)
        mode_sel = st.selectbox("Mode:", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"])

    # --- DASHBOARD: FIXED ANALYSIS LOGIC ---
    if nav == "🏠 Dashboard":
        st.title(f"Focus: {mode_sel}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card"><h4>1. CAPTURE</h4>', unsafe_allow_html=True)
            if st.button("🎤 START RECORDING"):
                try:
                    fs = 44100
                    rec = sd.rec(int(rec_dur * fs), samplerate=fs, channels=1, dtype='float32')
                    with st.status("Recording... Speak now!"): sd.wait()
                    write("speech.wav", fs, (rec * 32767).astype(np.int16))
                    st.success("Captured! Click Analyze below.")
                    st.rerun()
                except Exception as e: st.error(f"Mic Error: {e}")

            if os.path.exists("speech.wav"):
                st.audio("speech.wav")
                b_ana, b_del = st.columns(2)
                with b_ana:
                    if st.button("🛑 ANALYZE & STORE"):
                        with st.spinner("AI Evaluating..."):
                            model_w = load_whisper_model() # Fixed: Uses cached model
                            res = model_w.transcribe("speech.wav", language='en')
                            pace, energy, balance = get_full_analysis("speech.wav")
                            
                            # Saving to History
                            st.session_state.vocal_history.append({'Date': pd.Timestamp.now().strftime('%H:%M'), 'Pace': pace, 'Balance': balance, 'Mode': mode_sel})
                            st.session_state.update({'trans': res['text'], 'pace': pace, 'energy': energy, 'balance': balance, 'ready': True})
                            st.rerun()
                with b_del:
                    st.markdown('<div class="delete-container">', unsafe_allow_html=True)
                    if st.button("🗑️ DELETE"): # Robust file deletion
                        if os.path.exists("speech.wav"): os.remove("speech.wav")
                        st.session_state.ready = False
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card"><h4>2. FEEDBACK</h4>', unsafe_allow_html=True)
            if st.session_state.get('ready'):
                st.write(f"⏳ **Speed:** {st.session_state.pace} BPM")
                st.write(f"🔥 **Energy:** {st.session_state.energy}")
                st.write(f"⏱️ **Balance:** {st.session_state.balance}% Talk")
                st.success("Session stored successfully!")
            else: st.info("Record a session to see results.")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- PROGRESS PAGE (Stored History) ---
    elif nav == "📈 My Progress":
        st.title("Performance History")
        if not st.session_state.vocal_history:
            st.warning("No data found. Analyze a recording first!")
        else:
            df = pd.DataFrame(st.session_state.vocal_history)
            st.table(df) # Structured data table
            if st.button("🗑️ CLEAR ALL"):
                st.session_state.vocal_history = []
                st.rerun()

    elif nav == "🤖 AI Script Bot":
        st.title("AI Speech Writer")
        topic = st.text_input("Enter topic:")
        if st.button("Generate Script ✨"):
            res = ai_model.generate_content(f"Write a professional {mode_sel} script about: {topic}")
            st.markdown(f'<div class="metric-card">{res.text}</div>', unsafe_allow_html=True)