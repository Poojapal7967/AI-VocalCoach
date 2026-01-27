import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
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
# Stored history for progress tracking
if 'vocal_history' not in st.session_state: st.session_state.vocal_history = [] 

# --- 2. ELITE DARK UI STYLING ---
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
    /* Delete button specific style */
    .delete-container button {
        background: linear-gradient(90deg, #FF4B4B, #FF0000) !important;
    }
    .metric-card { background: #0D1117; border-radius: 20px; padding: 25px; border: 1px solid #30363D; margin-bottom: 20px; }
    .glass-card {
        background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px); border-radius: 24px; padding: 40px 20px; text-align: center;
    }
    .progress-card { background: #161B22; border-radius: 15px; padding: 20px; border-left: 5px solid #00A3FF; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. ANALYTICS ENGINE ---
def get_vocal_analysis(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    variation = float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    energy = "High Impact" if variation >= 18 else "Moderate" if variation >= 12 else "Monotone"
    intervals = librosa.effects.split(y, top_db=25)
    speech_time = sum([s[1] - s[0] for s in intervals]) / sr
    talk_ratio = (speech_time / (len(y)/sr)) * 100 if len(y) > 0 else 0
    return int(tempo[0] if isinstance(tempo, np.ndarray) else tempo), energy, int(talk_ratio)

# --- 4. NAVIGATION ENGINE ---
if st.session_state.page == 'home':
    st.markdown("""<div class="hero-box"><h1 style="font-size:4rem; font-weight:900;">Speak <span style="color: #6C5CE7;"><</span>Confidently.</h1></div>""", unsafe_allow_html=True)
    if st.button("🚀 ENTER DASHBOARD"):
        st.session_state.page = 'dashboard'
        st.rerun()
    st.write("##")
    f1, f2, f3 = st.columns(3)
    features = [("📊", "Real-time Metrics", "BPM & Pitch analysis."), ("🧠", "AI Script Bot", "Gemini powered scripts."), ("📈", "Progress", "Track growth charts.")]
    for col, (icon, title, desc) in zip([f1, f2, f3], features):
        col.markdown(f'<div class="glass-card"><h1>{icon}</h1><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

else:
    with st.sidebar:
        st.markdown("""<div style="background:linear-gradient(90deg, #FF00CC, #3333FF); padding:15px; border-radius:12px; text-align:center; margin-bottom:20px;"><h2 style="margin:0; color:white;">Speeko Elite</h2></div>""", unsafe_allow_html=True)
        if st.button("⬅️ HOME"):
            st.session_state.page = 'home'
            st.rerun()
        st.divider()
        nav = st.radio("Menu", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Script Bot"])
        rec_dur = st.slider("Rec duration (s):", 5, 60, 10)
        mode_sel = st.selectbox("Focus Mode:", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"])

    # --- DASHBOARD PAGE (Record/Store/Delete) ---
    if nav == "🏠 Dashboard":
        st.title(f"Coaching: {mode_sel}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card"><h4>1. RECORD PERFORMANCE</h4>', unsafe_allow_html=True)
            
            # Recording Logic
            if st.button(f"🎤 START RECORDING"):
                fs = 44100
                rec = sd.rec(int(rec_dur * fs), samplerate=fs, channels=1, dtype='float32')
                sd.wait(); write("speech.wav", fs, (rec * 32767).astype(np.int16))
                st.success("Recording captured!")
                st.rerun()
            
            # Storage & Deletion
            if os.path.exists("speech.wav"):
                st.audio("speech.wav")
                
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("🛑 ANALYZE & STORE"):
                        with st.spinner("AI analyzing..."):
                            model_w = whisper.load_model("base")
                            res = model_w.transcribe("speech.wav", language='en')
                            pace, energy, balance = get_vocal_analysis("speech.wav")
                            # Save to session history
                            st.session_state.vocal_history.append({
                                'Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                                'Mode': mode_sel,
                                'Pace': pace,
                                'Balance': balance
                            })
                            st.session_state.update({'trans': res['text'], 'pace': pace, 'energy': energy, 'balance': balance, 'ready': True})
                            st.rerun()
                
                with b2:
                    st.markdown('<div class="delete-container">', unsafe_allow_html=True)
                    if st.button("🗑️ DELETE RECORDING"):
                        os.remove("speech.wav")
                        st.session_state.ready = False
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card"><h4>2. AI INSIGHTS</h4>', unsafe_allow_html=True)
            if st.session_state.get('ready'):
                st.write(f"⏳ **Speed:** {st.session_state.pace} BPM")
                st.write(f"🔥 **Impact:** {st.session_state.energy}")
                st.write(f"⏱️ **Balance:** {st.session_state.balance}% Talk")
                st.success("Session saved to 'My Progress'!")
            else: st.info("Finish recording to see metrics.")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- PROGRESS PAGE (History Management) ---
    elif nav == "📈 My Progress":
        st.title("Your Stored Sessions")
        if not st.session_state.vocal_history:
            st.warning("Koi saved data nahi hai. Dashboard par recording karke analyze karein!")
        else:
            df = pd.DataFrame(st.session_state.vocal_history)
            
            # Deletion of all history
            if st.button("🗑️ CLEAR ALL PROGRESS"):
                st.session_state.vocal_history = []
                st.rerun()
                
            st.dataframe(df, use_container_width=True)
            
            st.markdown("### Growth Visualization")
            c_a, c_b = st.columns(2)
            with c_a:
                st.line_chart(df.set_index('Date')['Pace'])
            with c_b:
                st.bar_chart(df.set_index('Date')['Balance'])

    elif nav == "🤖 AI Script Bot":
        st.title("AI Speech Writer")
        topic = st.text_input("What is your topic?")
        if st.button("Generate Script ✨"):
            res = ai_model.generate_content(f"Write a professional {mode_sel} script about: {topic}")
            st.markdown(f'<div class="metric-card">{res.text}</div>', unsafe_allow_html=True)