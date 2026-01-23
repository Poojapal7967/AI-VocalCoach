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

# --- 1. PAGE ROUTING ---
if 'page' not in st.session_state: st.session_state.page = 'home'

# --- 2. SAME-TO-SAME UI STYLING ---
st.set_page_config(page_title="Speko AI", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #05070A; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    
    /* Hero Section */
    .hero-box {
        background: radial-gradient(circle at 70% 30%, rgba(108, 92, 231, 0.15) 0%, rgba(5, 7, 10, 1) 70%);
        padding: 80px 40px; border-radius: 40px; margin-bottom: 40px;
    }
    .hero-title { font-size: 4.2rem; font-weight: 900; letter-spacing: -2px; line-height: 1.1; margin-bottom: 20px; }
    
    /* Neon Pink/Blue Button */
    .stButton>button { 
        background: linear-gradient(90deg, #FF00CC, #3333FF); color: white; border-radius: 12px; 
        padding: 15px 35px; border: none; font-weight: 800; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0 0 30px rgba(255, 0, 204, 0.4); }

    /* Glass Cube Feature Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 40px 20px;
        text-align: center;
        transition: 0.4s;
        height: 100%;
    }
    .glass-card:hover { border-color: #FF00CC; transform: translateY(-10px); }
    
    /* Dashboard Polish */
    .metric-card { background: #0D1117; border-radius: 20px; padding: 25px; border: 1px solid #30363D; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. BACKEND ---
def get_advanced_stats(file_path):
    y, sr = librosa.load(file_path)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    variation = float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    intonation = "Dynamic" if variation >= 15 else "A bit Monotone"
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo), intonation, y, sr

# --- 4. NAVIGATION LOGIC (FIXED Syntax) ---
if st.session_state.page == 'home':
    st.markdown("""
        <div class="hero-box">
            <h1 class="hero-title">Speak <span style="color: #6C5CE7;"><</span>Confidently.</h1>
            <p style="font-size: 1.4rem; color: #8892B0; max-width: 600px; margin-bottom: 40px;">
                The AI speech coach that transforms voice with precision analytics.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 GET STARTED"):
        st.session_state.page = 'dashboard'
        st.rerun()

    st.write("##")
    st.markdown("<h3 style='text-align:center;'>Features</h3>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    features = [
        ("📊", "Real-time Insights", "Get instant feedback on pace and fillers."),
        ("🧠", "AI Practice Bot", "Generate scripts tailored for meetings."),
        ("📈", "Progress Tracking", "Visualize your growth over time.")
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3], features):
        col.markdown(f'<div class="glass-card"><h1>{icon}</h1><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

else:
    # --- DASHBOARD SECTION ---
    if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
    model_w = whisper.load_model("base")

    with st.sidebar:
        if st.button("⬅️ HOME"):
            st.session_state.page = 'home'
            st.rerun()
        st.divider()
        nav = st.radio("Menu", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Script Bot"])
        rec_dur = st.slider("Rec duration:", 3, 60, 5) #

    if nav == "🏠 Dashboard":
        st.title("Voice Performance")
        cl, cr = st.columns([1, 1])
        with cl:
            st.markdown('<div class="metric-card"><h4>1. RECORD</h4>', unsafe_allow_html=True)
            if st.button(f"🎤 START ({rec_dur}s)"):
                fs = 44100
                rec = sd.rec(int(rec_dur * fs), samplerate=fs, channels=1, dtype='float32')
                sd.wait(); write("speech.wav", fs, (rec * 32767).astype(np.int16)); st.rerun()
            if os.path.exists("speech.wav"):
                if st.button("🛑 ANALYZE"):
                    res = model_w.transcribe("speech.wav", language='en')
                    tempo, inton, y, sr = get_advanced_stats("speech.wav")
                    st.session_state.update({'trans': res['text'], 'tempo': tempo, 'inton': inton, 'ready': True})
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="metric-card"><h4>2. FEEDBACK</h4>', unsafe_allow_html=True)
            if st.session_state.get('ready'):
                st.write(f"**Speed:** {int(st.session_state.tempo)} BPM")
                st.write(f"**Energy:** {st.session_state.inton}")
            else: st.info("Waiting for recording...")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('ready'):
            st.markdown('<div class="metric-card"><h4>WAVEFORM</h4>', unsafe_allow_html=True)
            yp, srp = librosa.load("speech.wav")
            fig, ax = plt.subplots(figsize=(10, 2), facecolor='#0D1117')
            librosa.display.waveshow(yp, sr=srp, ax=ax, color='#00A3FF', alpha=0.8)
            ax.axis('off'); st.pyplot(fig) #
            st.write(f"**Transcript:** {st.session_state.trans}")
            st.markdown('</div>', unsafe_allow_html=True)

    elif nav == "🤖 AI Script Bot":
        st.title("AI script bot")
        # (Aapka existing bot code yahan bina error ke chalega)