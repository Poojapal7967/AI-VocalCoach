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
    st.error("Invalid API Key.")

# --- 1. STATE MANAGEMENT ---
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'vocal_history' not in st.session_state: st.session_state.vocal_history = [] 

# --- 2. ELITE DREAMSTIME UI ---
st.set_page_config(page_title="Speeko Elite", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #05070A; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    .hero-box {
        background: radial-gradient(circle at 70% 30%, rgba(108, 92, 231, 0.15) 0%, rgba(5, 7, 10, 1) 70%);
        padding: 60px 40px; border-radius: 30px; margin-bottom: 30px; text-align: center;
    }
    .hero-title { font-size: 4rem; font-weight: 900; letter-spacing: -2px; line-height: 1.1; margin-bottom: 20px; }
    .stButton>button { 
        background: linear-gradient(90deg, #FF00CC, #3333FF); color: white; border-radius: 12px; 
        padding: 12px 30px; border: none; font-weight: 800; transition: 0.3s; width: 100%;
    }
    .metric-card { background: #0D1117; border-radius: 20px; padding: 25px; border: 1px solid #30363D; margin-bottom: 20px; }
    .progress-card { background: #161B22; border-radius: 15px; padding: 20px; border-left: 5px solid #00A3FF; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. CORE BACKEND (FULL LOGIC) ---
def get_full_analysis(file_path):
    y, sr = librosa.load(file_path)
    # 1. Pace (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # 2. Energy (Pitch Variation)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    variation = float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    energy_label = "High Impact" if variation >= 18 else "Moderate" if variation >= 12 else "Monotone"
    # 3. Speech Ratio (Talk Time)
    intervals = librosa.effects.split(y, top_db=25)
    speech_time = sum([s[1] - s[0] for s in intervals]) / sr
    talk_ratio = (speech_time / (len(y)/sr)) * 100 if len(y) > 0 else 0
    return int(tempo[0] if isinstance(tempo, np.ndarray) else tempo), energy_label, int(talk_ratio), y, sr

def get_smart_feedback(mode, transcript):
    # Filler word logic
    fillers = ["um", "ah", "uh", "like", "basically", "actually"]
    found = [w for w in transcript.lower().split() if w in fillers]
    
    # Mode-based Coaching tips
    tips = {
        "🎧 Anchoring": "🎙️ Focus on resonance and smooth transitions.",
        "👨‍🏫 Teaching": "🍎 Use deliberate pauses to emphasize key facts.",
        "🎤 Public Speaking": "🌟 Project your voice and use emotional pitch variation.",
        "💼 Interview": "🤝 Match your energy to the company culture."
    }
    return tips.get(mode, "Great effort!"), len(found)

# --- 4. NAVIGATION ENGINE ---
if st.session_state.page == 'home':
    st.markdown('<div class="hero-box"><h1 class="hero-title">Speak <span style="color: #6C5CE7;"><</span>Confidently.</h1><p>Master Every Mode of Communication</p></div>', unsafe_allow_html=True)
    if st.button("🚀 ENTER DASHBOARD"):
        st.session_state.page = 'dashboard'
        st.rerun()
else:
    # --- APP SIDEBAR ---
    with st.sidebar:
        if st.button("⬅️ HOME"):
            st.session_state.page = 'home'
            st.rerun()
        st.divider()
        nav = st.radio("Navigate", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Bot"])
        rec_dur = st.slider("Record Duration (s):", 5, 60, 10)
        mode_sel = st.selectbox("Coaching Mode:", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"])

    if nav == "🏠 Dashboard":
        st.title(f"Focus: {mode_sel}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card"><h4>1. CAPTURE</h4>', unsafe_allow_html=True)
            if st.button(f"🎤 START RECORDING"):
                fs = 44100
                rec = sd.rec(int(rec_dur * fs), samplerate=fs, channels=1, dtype='float32')
                sd.wait(); write("speech.wav", fs, (rec * 32767).astype(np.int16)); st.rerun()
            if os.path.exists("speech.wav"):
                st.audio("speech.wav")
                if st.button("🛑 ANALYZE"):
                    with st.spinner("AI Processing..."):
                        model_w = whisper.load_model("base")
                        res = model_w.transcribe("speech.wav", language='en')
                        pace, energy, balance, y, sr = get_full_analysis("speech.wav")
                        tip, f_count = get_smart_feedback(mode_sel, res['text'])
                        
                        # Save for Progress Tracking
                        st.session_state.vocal_history.append({'Time': pd.Timestamp.now().strftime('%H:%M'), 'Pace': pace, 'Balance': balance, 'Mode': mode_sel})
                        st.session_state.update({'trans': res['text'], 'pace': pace, 'energy': energy, 'balance': balance, 'tip': tip, 'fillers': f_count, 'ready': True})
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card"><h4>2. AI REPORT</h4>', unsafe_allow_html=True)
            if st.session_state.get('ready'):
                st.write(f"⏳ **Speed:** {st.session_state.pace} BPM")
                st.write(f"🔥 **Impact:** {st.session_state.energy}")
                st.write(f"⏱️ **Balance:** {st.session_state.balance}% Talk")
                st.write(f"💬 **Fillers:** {st.session_state.fillers} detected")
                st.success(st.session_state.tip)
            else: st.info("Finish recording to see metrics.")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('ready'):
            st.markdown('<div class="metric-card"><h4>WAVEFORM ANALYSIS</h4>', unsafe_allow_html=True)
            yp, srp = librosa.load("speech.wav")
            fig, ax = plt.subplots(figsize=(10, 2), facecolor='#0D1117')
            librosa.display.waveshow(yp, sr=srp, ax=ax, color='#FF00CC', alpha=0.8)
            ax.axis('off'); st.pyplot(fig) #
            st.write(f"**Transcript:** {st.session_state.trans}")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- PROGRESS PAGE (FULL CHARTING) ---
    elif nav == "📈 My Progress":
        st.title("Performance Growth")
        if not st.session_state.vocal_history:
            st.info("Record sessions to generate your growth charts!")
        else:
            df = pd.DataFrame(st.session_state.vocal_history)
            ca, cb = st.columns(2)
            with ca:
                st.markdown('<div class="progress-card"><h4>Pace Trend (BPM)</h4></div>', unsafe_allow_html=True)
                st.line_chart(df.set_index('Time')['Pace'])
            with cb:
                st.markdown('<div class="progress-card"><h4>Engagement Balance (%)</h4></div>', unsafe_allow_html=True)
                st.bar_chart(df.set_index('Time')['Balance'])
            st.table(df)

    elif nav == "🤖 AI Bot":
        st.title("AI Speech Writer")
        topic = st.text_input("Enter your speech topic:")
        if st.button("Generate Script ✨"):
            res = ai_model.generate_content(f"Write a {mode_sel} script about: {topic}")
            st.markdown(f'<div class="metric-card">{res.text}</div>', unsafe_allow_html=True)