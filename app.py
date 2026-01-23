import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import google.generativeai as genai 

# --- 0. AI CONFIG ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-pro')
except:
    st.error("Invalid API Key.")

# --- 1. PAGE STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- 2. ELITE DARK UI STYLING ---
st.set_page_config(page_title="Speeko AI | Elite Coach", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0B0E14; color: #E2E8F0; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0F1219; border-right: 1px solid #1E293B; }
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(12px); padding: 80px 20px; border-radius: 40px;
        text-align: center; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 50px;
    }
    .hero-title { font-size: 4.5rem; font-weight: 900; color: #FFFFFF; letter-spacing: -3px; line-height: 1; }
    .neon-text { color: #00A3FF; text-shadow: 0 0 20px rgba(0, 163, 255, 0.5); }
    .elite-card {
        background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 28px; padding: 40px 25px; text-align: center; transition: 0.4s; height: 100%;
    }
    .elite-card:hover { border-color: #00A3FF; transform: translateY(-10px); }
    .metric-card { background: #161B22; border-radius: 20px; padding: 25px; border: 1px solid #30363D; margin-bottom: 20px; }
    .stButton>button { 
        background: linear-gradient(90deg, #00A3FF, #0066FF); color: white; border-radius: 12px; 
        padding: 14px 28px; border: none; font-weight: 700; width: 100%; transition: 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. BACKEND LOGIC ---
def get_advanced_stats(file_path):
    y, sr = librosa.load(file_path)
    intervals = librosa.effects.split(y, top_db=25)
    speech_time = sum([s[1] - s[0] for s in intervals]) / sr
    talk_ratio = (speech_time / (len(y)/sr)) * 100 if len(y) > 0 else 0
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    variation = float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    intonation = "Dynamic & Energetic" if variation >= 15 else "A bit Monotone"
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo), variation, intonation, talk_ratio, y, sr

# --- 4. MAIN ROUTING ENGINE ---
if st.session_state.page == 'home':
    # LANDING PAGE
    st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">Speak <span class="neon-text">Confidently.</span></h1>
            <p style="font-size: 1.3rem; color: #94A3B8; max-width: 750px; margin: 20px auto 40px auto;">
                The elite AI speech coach. Master your tone, eliminate fillers, and refine your voice.
            </p>
        </div>
    """, unsafe_allow_html=True)
    _, col_btn, _ = st.columns([1, 0.7, 1])
    with col_btn:
        if st.button("🚀 ENTER DASHBOARD"):
            st.session_state.page = 'dashboard'
            st.rerun()
    st.write("##")
    f1, f2, f3 = st.columns(3)
    features = [("📊", "Metrics", "BPM & Pitch analysis."), ("🧠", "AI Coach", "Gemini powered scripts."), ("🌊", "Waveform", "Real-time visualizer.")]
    for col, (icon, title, desc) in zip([f1, f2, f3], features):
        col.markdown(f'<div class="elite-card"><h2>{icon}</h2><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

else:
    # --- DASHBOARD / APP SECTION ---
    if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    
    # Load model only when in dashboard to save resources
    @st.cache_resource
    def load_whisper(): return whisper.load_model("base")
    model_w = load_whisper()

    with st.sidebar:
        if st.button("⬅️ EXIT TO HOME"):
            st.session_state.page = 'home'
            st.rerun()
        st.divider()
        nav = st.radio("Menu", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Script Bot"])
        rec_duration = st.slider("Rec duration (s):", 3, 60, 5)
        mode_sel = st.radio("Focus Mode:", ["🎤 Public Speaking", "💼 Interview", "👨‍🏫 Teaching"])

    # FIXED NAVIGATION NESTING
    if nav == "🏠 Dashboard":
        st.title(f"Refining: {mode_sel}")
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown('<div class="metric-card"><p style="font-weight:700; color:#00A3FF;">1. CAPTURE</p>', unsafe_allow_html=True)
            if st.button(f"🎤 START RECORDING ({rec_duration}s)"):
                fs = 44100
                rec = sd.rec(int(rec_duration * fs), samplerate=fs, channels=1, dtype='float32')
                sd.wait(); write("speech.wav", fs, (rec * 32767).astype(np.int16)); st.rerun()
            if os.path.exists("speech.wav"):
                st.audio("speech.wav")
                if st.button("🛑 ANALYZE CLIP"):
                    res = model_w.transcribe("speech.wav", fp16=False, language='en')
                    tempo, _, inton, talk, y, sr = get_advanced_stats("speech.wav")
                    st.session_state.update({'transcription': res['text'], 'tempo': tempo, 'intonation': inton, 'talk_ratio': talk, 'analysis_ready': True})
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="metric-card"><p style="font-weight:700; color:#00A3FF;">2. INSIGHTS</p>', unsafe_allow_html=True)
            if st.session_state.get('analysis_ready'):
                st.write(f"**Speed:** {int(st.session_state.tempo)} BPM")
                st.write(f"**Energy:** {st.session_state.intonation}")
                st.write(f"**Talk Time:** {int(st.session_state.talk_ratio)}%")
            else: st.info("Waiting for data...")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('analysis_ready'):
            st.markdown('<div class="metric-card"><p style="font-weight:700; color:#00A3FF;">HEATMAP</p>', unsafe_allow_html=True)
            yp, srp = librosa.load("speech.wav")
            fig, ax = plt.subplots(figsize=(12, 2), facecolor='#161B22')
            librosa.display.waveshow(yp, sr=srp, ax=ax, color='#00A3FF', alpha=0.8)
            ax.set_facecolor('#161B22'); ax.axis('off'); st.pyplot(fig) #
            st.markdown('</div>', unsafe_allow_html=True)

    elif nav == "📈 My Progress":
        st.title("Performance History")
        st.info("Metrics tracker coming soon.")

    elif nav == "🤖 AI Script Bot":
        st.title("Elite Practice Scripts")
        t_in = st.text_input("What's the topic?")
        if st.button("Generate Script ✨"):
            if t_in:
                response = ai_model.generate_content(f"Write a practice script for {mode_sel}: {t_in}")
                st.session_state.chat_history.append({"a": response.text}); st.rerun()
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f'<div class="metric-card" style="border-left: 5px solid #00A3FF;">{chat["a"]}</div>', unsafe_allow_html=True)