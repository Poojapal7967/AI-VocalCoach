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

# --- 0. AI CHATBOT CONFIG ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-pro')
except:
    st.error("Invalid API Key. Update Gemini Key in code.")

# --- 1. PAGE STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- 2. UI STYLING (Speeko Professional) ---
st.set_page_config(page_title="Speeko AI Coach", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #F7F9FC; color: #333333; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E6ED; }
    .metric-card { background: white; border-radius: 20px; padding: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #E0E6ED; margin-bottom: 25px; }
    .card-title { font-size: 1.4rem; font-weight: 700; color: #111827; margin-bottom: 15px; }
    .stButton>button { background-color: #00A3FF; color: white; border-radius: 50px; padding: 15px 30px; border: none; font-weight: 800; font-size: 18px; width: 100%; box-shadow: 0 4px 10px rgba(0, 163, 255, 0.3); }
    .feedback-row { display: flex; align-items: center; padding: 15px 0; border-bottom: 1px solid #F0F2F6; }
    .icon-box { width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 20px; }
    
    /* Hero Section Styling */
    .hero-text { text-align: center; padding: 60px 0; }
    .hero-title { font-size: 3.5rem; font-weight: 800; color: #111827; margin-bottom: 10px; }
    .hero-sub { font-size: 1.2rem; color: #4B5563; max-width: 800px; margin: 0 auto 30px auto; }
    </style>
""", unsafe_allow_html=True)

# --- 3. BACKEND LOGIC ---
def get_advanced_stats(file_path):
    y, sr = librosa.load(file_path)
    intervals = librosa.effects.split(y, top_db=25)
    speech_time = sum([start_end[1] - start_end[0] for start_end in intervals]) / sr
    total_time = len(y) / sr
    talk_ratio = (speech_time / total_time) * 100 if total_time > 0 else 0
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    variation = float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    intonation = "Dynamic & Energetic" if variation >= 15 else "A bit Monotone"
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)): tempo = float(tempo[0])
    return tempo, np.mean(pitches[pitches > 0]), intonation, talk_ratio, y, sr

def get_smart_feedback(mode, transcript, tempo):
    fillers = ["um", "ummm", "ah", "ahh", "uh", "uhh", "basically", "actually", "like", "you know"]
    words = transcript.lower().replace(".", "").replace(",", "").split()
    found = [w for w in words if w in fillers]
    tips = ["🌟 **Impact:** Use volume variety to capture audience focus."]
    if mode == "💼 Interview": tips = ["🤝 **Pro Tip:** End your sentences firmly to show authority."]
    return tips, found

# --- 4. NAVIGATION LOGIC ---
if st.session_state.page == 'home':
    # FRONTEND LANDING PAGE
    st.markdown('<div class="hero-text">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Speak confidently.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Master your presentations, ace your interviews, and find your perfect voice with our AI-powered speech coach.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    _, col_btn, _ = st.columns([1, 0.6, 1])
    with col_btn:
        if st.button("🚀 Start Free Coaching Session"):
            st.session_state.page = 'dashboard'
            st.rerun()
            
    st.write("##")
    st.divider()
    
    # Feature Showcase
    f1, f2, f3 = st.columns(3)
    f1.markdown('<div class="metric-card"><h3>📊 Real-time Insights</h3><p>Instant feedback on pace, tone, and fillers.</p></div>', unsafe_allow_html=True)
    f2.markdown('<div class="metric-card"><h3>🤖 AI Scripts</h3><p>Generate practice scripts for any scenario.</p></div>', unsafe_allow_html=True)
    f3.markdown('<div class="metric-card"><h3>📈 Progress Tracking</h3><p>Watch your confidence grow over time.</p></div>', unsafe_allow_html=True)

else:
    # --- DASHBOARD PAGE ---
    if 'vocal_history' not in st.session_state: st.session_state.vocal_history = [] 
    if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    model_w = whisper.load_model("base")

    with st.sidebar:
        if st.button("⬅️ Back to Home"):
            st.session_state.page = 'home'
            st.rerun()
        st.divider()
        st.markdown("<h2 style='color:#00A3FF; font-weight:900;'>Speeko AI</h2>", unsafe_allow_html=True)
        nav = st.radio("Menu", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Script Bot"])
        st.divider()
        rec_duration = st.slider("Set recording duration (s):", 3, 60, 5)
        mode_sel = st.radio("Focus Mode:", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"])

    if nav == "🏠 Dashboard":
        st.title(f"Ready to master your {mode_sel.split()[-1]} skills?")
        col_l, col_r = st.columns([1, 1])
        
        with col_l:
            st.markdown('<div class="metric-card"><p class="card-title">1. Record Your Voice</p>', unsafe_allow_html=True)
            if st.button(f"🎤 START RECORDING ({rec_duration}s)"):
                with st.status("Listening..."):
                    fs = 44100
                    rec = sd.rec(int(rec_duration * fs), samplerate=fs, channels=1, dtype='float32')
                    sd.wait()
                    write("speech.wav", fs, (rec * 32767).astype(np.int16))
                    st.rerun()
            if os.path.exists("speech.wav"):
                st.audio("speech.wav")
                if st.button("🛑 ANALYZE MY VOICE"):
                    with st.spinner("AI analyzing..."):
                        res = model_w.transcribe("speech.wav", fp16=False, language='en')
                        tempo, pitch, inton, talk, y, sr = get_advanced_stats("speech.wav")
                        tips, found = get_smart_feedback(mode_sel, res['text'], tempo)
                        word_count = len(res['text'].split())
                        f_perc = int((len(found) / word_count) * 100) if word_count > 0 else 0
                        st.session_state.vocal_history.append({'score': 85, 'fillers': f_perc, 'talk': talk})
                        st.session_state.update({'transcription': res['text'], 'tips': tips, 'filler_count': f_perc, 'tempo': tempo, 'intonation': inton, 'talk_ratio': talk, 'analysis_ready': True})
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="metric-card"><p class="card-title">2. Your AI Feedback</p>', unsafe_allow_html=True)
            if st.session_state.get('analysis_ready'):
                perf_data = [
                    ("Talking Speed", f"{int(st.session_state.tempo)} BPM", "⏳", "#FFF4ED"),
                    ("Word Clarity", f"{st.session_state.filler_count}% fillers", "💬", "#EBF5FF"),
                    ("Voice Energy", st.session_state.intonation, "🎵", "#E9FBF0"),
                    ("Speech Balance", f"{int(st.session_state.talk_ratio)}% Talk Time", "⏱️", "#FFF0F0")
                ]
                for l, v, i, b in perf_data:
                    st.markdown(f'<div class="feedback-row"><div class="icon-box" style="background-color: {b};">{i}</div><div style="flex-grow: 1;"><div style="font-size: 11px; color: #8898AA;">{l}</div><div style="font-weight: 700; font-size: 16px;">{v}</div></div></div>', unsafe_allow_html=True)
            else: st.info("Record to see insights.")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('analysis_ready'):
            st.markdown('<div class="metric-card"><p class="card-title">Confidence Visualization</p>', unsafe_allow_html=True)
            yp, srp = librosa.load("speech.wav")
            fig, ax = plt.subplots(figsize=(12, 2))
            h_color = '#00A3FF' if "Dynamic" in st.session_state.intonation else '#FF4B4B'
            librosa.display.waveshow(yp, sr=srp, ax=ax, color=h_color, alpha=0.8)
            ax.axis('off'); st.pyplot(fig)
            st.markdown(f'**Transcript:** "{st.session_state.transcription}"')
            for tip in st.session_state.tips: st.write(f"💡 {tip}")
            st.markdown('</div>', unsafe_allow_html=True)

    elif nav == "📈 My Progress":
        st.title("Track your speaker progress.")
        if st.session_state.vocal_history:
            st.line_chart([s['talk'] for s in st.session_state.vocal_history])
        else: st.info("Need sessions to see trends.")

    elif nav == "🤖 AI Script Bot":
        st.title("Personalized practice scripts.")
        t_in = st.text_input("Enter topic:")
        if st.button("Generate Script ✨"):
            if t_in:
                response = ai_model.generate_content(f"Write a script for {mode_sel}: {t_in}")
                st.session_state.chat_history.append({"a": response.text}); st.rerun()
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f'<div class="metric-card" style="border-left: 5px solid #00A3FF;">{chat["a"]}</div>', unsafe_allow_html=True)