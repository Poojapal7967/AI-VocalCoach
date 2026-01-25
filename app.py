import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import pandas as pd # Performance tracking ke liye
import google.generativeai as genai 

# --- 0. AI CONFIG ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-pro')
except:
    st.error("Invalid API Key.")

# --- 1. PAGE ROUTING & HISTORY ---
if 'page' not in st.session_state: st.session_state.page = 'home'
# Persistent history for tracking
if 'vocal_history' not in st.session_state: st.session_state.vocal_history = [] 

# --- 2. SAME-TO-SAME UI STYLING ---
st.set_page_config(page_title="Speeko AI", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #05070A; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    .hero-box {
        background: radial-gradient(circle at 70% 30%, rgba(108, 92, 231, 0.15) 0%, rgba(5, 7, 10, 1) 70%);
        padding: 80px 40px; border-radius: 40px; margin-bottom: 40px;
    }
    .hero-title { font-size: 4.2rem; font-weight: 900; letter-spacing: -2px; line-height: 1.1; margin-bottom: 20px; }
    .stButton>button { 
        background: linear-gradient(90deg, #FF00CC, #3333FF); color: white; border-radius: 12px; 
        padding: 15px 35px; border: none; font-weight: 800; transition: 0.3s;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px); border-radius: 24px; padding: 40px 20px; text-align: center;
    }
    .metric-card { background: #0D1117; border-radius: 20px; padding: 25px; border: 1px solid #30363D; margin-bottom: 20px; }
    .progress-card { background: #161B22; border-radius: 15px; padding: 20px; border-left: 5px solid #00A3FF; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. BACKEND ---
def get_advanced_stats(file_path):
    y, sr = librosa.load(file_path)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    variation = float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    intonation = "Dynamic" if variation >= 15 else "A bit Monotone"
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # Speech balance calculation
    intervals = librosa.effects.split(y, top_db=25)
    speech_time = sum([s[1] - s[0] for s in intervals]) / sr
    talk_ratio = (speech_time / (len(y)/sr)) * 100 if len(y) > 0 else 0
    return float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo), intonation, talk_ratio, y, sr

# --- 4. NAVIGATION LOGIC (FIXED Syntax) ---
if st.session_state.page == 'home':
    st.markdown("""<div class="hero-box"><h1 class="hero-title">Speak <span style="color: #6C5CE7;"><</span>Confidently.</h1></div>""", unsafe_allow_html=True)
    if st.button("🚀 GET STARTED"):
        st.session_state.page = 'dashboard'
        st.rerun()
    st.write("##")
    f1, f2, f3 = st.columns(3)
    features = [("📊", "Insights", "Real-time analysis."), ("🧠", "AI Bot", "Practice scripts."), ("📈", "Progress", "Track growth.")]
    for col, (icon, title, desc) in zip([f1, f2, f3], features):
        col.markdown(f'<div class="glass-card"><h1>{icon}</h1><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

else:
    # --- APP SIDEBAR ---
    with st.sidebar:
        if st.button("⬅️ HOME"):
            st.session_state.page = 'home'
            st.rerun()
        st.divider()
        nav = st.radio("Menu", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Script Bot"])
        rec_dur = st.slider("Rec duration:", 3, 60, 5)

    # --- DASHBOARD PAGE ---
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
                    model_w = whisper.load_model("base")
                    res = model_w.transcribe("speech.wav", language='en')
                    tempo, inton, talk, y, sr = get_advanced_stats("speech.wav")
                    # Data ko history mein save karna
                    st.session_state.vocal_history.append({
                        'Time': pd.Timestamp.now().strftime('%H:%M:%S'),
                        'Pace': int(tempo),
                        'Balance': int(talk)
                    })
                    st.session_state.update({'trans': res['text'], 'tempo': tempo, 'inton': inton, 'talk': talk, 'ready': True})
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="metric-card"><h4>2. FEEDBACK</h4>', unsafe_allow_html=True)
            if st.session_state.get('ready'):
                # User-friendly labels
                st.write(f"⏳ **Talking Speed:** {int(st.session_state.tempo)} BPM")
                st.write(f"🎵 **Voice Energy:** {st.session_state.inton}")
                st.write(f"⏱️ **Speech Balance:** {int(st.session_state.talk)}% Talk Time")
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

    # --- MY PROGRESS PAGE (NEWLY INTEGRATED) ---
    elif nav == "📈 My Progress":
        st.title("Your Speaking Growth")
        if not st.session_state.vocal_history:
            st.info("Dashboard par jaakar apni voice analyze karein taaki yahan data show ho sake!")
        else:
            df = pd.DataFrame(st.session_state.vocal_history)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="progress-card"><h4>Pace Trend (BPM)</h4></div>', unsafe_allow_html=True)
                st.line_chart(df.set_index('Time')['Pace'])
            with c2:
                st.markdown('<div class="progress-card"><h4>Talk Time Balance (%)</h4></div>', unsafe_allow_html=True)
                st.bar_chart(df.set_index('Time')['Balance']) #
            
            st.write("### Session Log")
            st.table(df)

    elif nav == "🤖 AI Script Bot":
        st.title("AI Script Bot")
        t_in = st.text_input("Topic for practice script:")
        if st.button("Generate ✨"):
            if t_in:
                response = ai_model.generate_content(f"Write a short practice script for: {t_in}")
                st.markdown(f'<div class="metric-card">{response.text}</div>', unsafe_allow_html=True)