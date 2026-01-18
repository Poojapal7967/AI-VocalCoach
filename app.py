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
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # Apni key yahan daalein
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-pro')
except:
    st.error("Invalid API Key. Update Gemini Key in code.")

# --- 1. SPEEKO-STYLE UI (Full Professional Styling) ---
st.set_page_config(page_title="Speeko AI Coach", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #F7F9FC; color: #333333; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E6ED; }
    .metric-card { background: white; border-radius: 20px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #E0E6ED; margin-bottom: 20px; }
    .metric-label { font-size: 13px; color: #8898AA; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .stButton>button { background-color: #00A3FF; color: white; border-radius: 50px; padding: 12px 25px; border: none; font-weight: bold; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background-color: #0082CC; transform: translateY(-2px); }
    .feedback-row { display: flex; align-items: center; padding: 12px 0; border-bottom: 1px solid #F0F2F6; }
    .icon-box { width: 35px; height: 35px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. ADVANCED BACKEND LOGIC (Full Metrics + Heatmap + Talk-Time) ---
def get_advanced_stats(file_path):
    y, sr = librosa.load(file_path)
    
    # Talk Time Balance (Silence vs Speech)
    intervals = librosa.effects.split(y, top_db=25)
    speech_time = sum([start_end[1] - start_end[0] for start_end in intervals]) / sr
    total_time = len(y) / sr
    talk_ratio = (speech_time / total_time) * 100 if total_time > 0 else 0

    # Pitch & Intonation Logic
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
    variation = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
    intonation = "Monotone" if variation < 15 else "Dynamic"
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)): tempo = float(tempo[0])
    
    return tempo, avg_pitch, intonation, talk_ratio, y, sr

def get_smart_feedback(mode, transcript, tempo):
    fillers = ["um", "ah", "basically", "actually", "like"]
    words = transcript.lower().split()
    found = [w for w in words if w in fillers]
    tips = []
    if mode == "💼 Interview": tips.append("🤝 **Pro Tip:** Authority comes from calm, steady finishes.")
    elif mode == "👨‍🏫 Teaching": tips.append("📖 **Educator Hack:** Slow down on technical terms.")
    else: tips.append("🌟 **Impact:** Use volume variety to capture audience focus.")
    if len(found) > 0: tips.append(f"⚠️ **Clarity:** Found {len(found)} filler words.")
    else: tips.append("✅ **Clarity:** Exceptional speech flow.")
    return tips, found

# --- 3. SESSION INITIALIZATION ---
if 'vocal_history' not in st.session_state: st.session_state.vocal_history = [] 
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
model_w = whisper.load_model("base")

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='color:#00A3FF; font-weight:800;'>Speeko AI</h2>", unsafe_allow_html=True)
    nav = st.radio("Menu", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Script Bot"])
    st.divider()
    st.markdown("**Focus Mode:**")
    mode_sel = st.radio("", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"])

# --- 5. MAIN DASHBOARD ---
if nav == "🏠 Dashboard":
    st.title("Feedback just for you.") #
    cl, cr = st.columns([1.2, 1])
    
    with cl: # Audio Box
        st.markdown('<div class="metric-card"><p class="metric-label">Voice Recorder</p>', unsafe_allow_html=True)
        if os.path.exists("speech.wav"): st.audio("speech.wav")
        
        b_c1, b_c2 = st.columns(2)
        if b_c1.button("🎤 START RECORDING"):
            with st.status("Recording..."):
                rec = sd.rec(int(5 * 44100), samplerate=44100, channels=1, dtype='float32')
                sd.wait()
                write("speech.wav", 44100, (rec * 32767).astype(np.int16))
                st.rerun()
        
        if b_c2.button("🛑 ANALYZE"):
            if os.path.exists("speech.wav"):
                with st.spinner("Analyzing..."):
                    res = model_w.transcribe("speech.wav")
                    tempo, pitch, inton, talk, y, sr = get_advanced_stats("speech.wav")
                    tips, found = get_smart_feedback(mode_sel, res['text'], tempo)
                    f_perc = int((len(found) / len(res['text'].split())) * 100) if len(res['text'].split()) > 0 else 0
                    
                    st.session_state.vocal_history.append({'score': 85, 'fillers': f_perc, 'talk': talk})
                    st.session_state.update({
                        'transcription': res['text'], 'tips': tips, 'filler_count': f_perc, 
                        'tempo': tempo, 'avg_pitch': pitch, 'intonation': inton, 
                        'talk_ratio': talk, 'analysis_ready': True
                    })
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with cr: # Professional Metrics
        st.markdown('<div class="metric-card"><p class="metric-label">Vocal Performance</p>', unsafe_allow_html=True)
        perf_data = [
            ("Pace", f"{int(st.session_state.get('tempo', 0))} BPM", "⏳", "#FFF4ED"),
            ("Talk Time", f"{int(st.session_state.get('talk_ratio', 0))}% Speech", "⏱️", "#EBF5FF"), #
            ("Intonation", st.session_state.get('intonation', 'Pending'), "🎵", "#E9FBF0"), #
            ("Pitch (Avg)", f"{int(st.session_state.get('avg_pitch', 0))} Hz", "🔊", "#FFF0F0")
        ]
        for l, v, i, b in perf_data:
            st.markdown(f'<div class="feedback-row"><div class="icon-box" style="background-color: {b};">{i}</div><div style="flex-grow: 1;"><div style="font-size: 11px; color: #8898AA;">{l}</div><div style="font-weight: 700; font-size: 16px;">{v}</div></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get('analysis_ready'):
        # Confidence Heatmap
        st.markdown('<div class="metric-card"><p class="metric-label">Confidence Heatmap</p>', unsafe_allow_html=True)
        yp, srp = librosa.load("speech.wav")
        fig, ax = plt.subplots(figsize=(12, 2.5))
        
        # Color logic based on modulation
        h_color = '#00A3FF' if st.session_state.intonation == "Dynamic" else '#FF4B4B'
        librosa.display.waveshow(yp, sr=srp, ax=ax, color=h_color, alpha=0.8)
        ax.axis('off'); st.pyplot(fig)
        st.markdown(f'<em>"{st.session_state.transcription}"</em></div>', unsafe_allow_html=True)

elif nav == "📈 My Progress": # History
    st.title("Track your speaker progress.")
    if len(st.session_state.vocal_history) > 1:
        prev, now = st.session_state.vocal_history[-2], st.session_state.vocal_history[-1]
        st.metric("Talk Time Balance", f"{int(now['talk'])}%", f"{int(now['talk'] - prev['talk'])}% change")
    else: st.info("Need 2 sessions to see trends.")

elif nav == "🤖 AI Script Bot": # Bot
    st.title("Personalized practice.")
    topic = st.text_input("What to practice today?")
    if st.button("Generate Script ✨"):
        if topic:
            response = ai_model.generate_content(f"Script for {mode_sel}: {topic}")
            st.session_state.chat_history.append({"a": response.text}); st.rerun()
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f'<div class="metric-card" style="border-left: 5px solid #00A3FF;">{chat["a"]}</div>', unsafe_allow_html=True)