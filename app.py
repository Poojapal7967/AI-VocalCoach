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
# Yahan apni Gemini API Key paste karein
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" 
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-pro')
except:
    st.error("Invalid API Key. Update Gemini Key in code.")

# --- 1. PREMIUM NEON UI (NO REDUCTIONS) ---
st.set_page_config(page_title="AI Vocal Coach Pro", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #08081a; color: #ffffff; font-family: 'Inter', sans-serif; }
    .recording-dot { width: 14px; height: 14px; background: #ff4b4b; border-radius: 50%; display: inline-block; margin-right: 10px; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
    .neon-card { background: rgba(255, 255, 255, 0.04); border-radius: 25px; padding: 40px; text-align: center; height: 350px; border: 1px solid rgba(255, 255, 255, 0.1); }
    .big-num { font-size: 80px; font-weight: 900; line-height: 1; margin-bottom: 10px; }
    .label-sub { font-size: 14px; color: #a0a0b0; text-transform: uppercase; letter-spacing: 2px; }
    .feature-box { background: rgba(0, 242, 254, 0.05); border-radius: 15px; padding: 15px; border-left: 4px solid #00f2fe; margin-top: 10px; text-align: left; }
    .highlight { color: #ff4b4b; font-weight: bold; text-decoration: underline; }
    div.stAudio { margin-top: 15px; border-radius: 50px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. BACKEND LOGIC (TypeError Fixed) ---
def get_vocal_stats(file_path):
    y, sr = librosa.load(file_path)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    avg_pitch = float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)): tempo = float(tempo[0])
    return tempo, avg_pitch, y, sr

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

# --- 3. SESSION & MODES ---
if 'mode' not in st.session_state: st.session_state.mode = "🎤 Public Speaking"
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
model_w = whisper.load_model("base")

st.markdown('<p style="color:#00f2fe; font-weight:800; font-size:24px;">🎯 MODE SELECTOR</p>', unsafe_allow_html=True)
st.session_state.mode = st.radio("", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"], horizontal=True)

# --- 4. DASHBOARD GRID ---
st.write("##")
c1, c2, c3 = st.columns(3)

with c1: # Score Card
    score = st.session_state.get('conf_score', 85)
    st.markdown(f'<div class="neon-card"><div class="label-sub">CONFIDENCE SCORE</div><div class="big-num">{score}</div><div style="height:8px; background:rgba(255,255,255,0.1); border-radius:10px; width:80%; margin:20px auto;"><div style="height:100%; width:{score}%; background:linear-gradient(to right, #00f2fe, #4caf50); border-radius:10px;"></div></div><div style="color:#4caf50; font-size:12px;">😊 SENTIMENT: POSITIVE</div></div>', unsafe_allow_html=True)

with c2: # Center Card Fix
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    if os.path.exists("speech.wav"):
        st.markdown('<div style="font-size:40px">🎙️</div>', unsafe_allow_html=True)
        st.audio("speech.wav")
        st.markdown(f'<div class="label-sub">{st.session_state.mode} READY</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="label-sub">NO DATA FOUND</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3: # Filler Card
    st.markdown(f'<div class="neon-card"><div class="big-num">{st.session_state.get("filler_count", 0)}%</div><div class="label-sub">Fillers Found</div></div>', unsafe_allow_html=True)

# --- 5. CONTROLS ---
st.write("##")
_, cb, _ = st.columns([1, 2, 1])
with cb:
    b1, b2 = st.columns(2)
    if b1.button("🎤 START RECORDING"):
        rec = sd.rec(int(5 * 44100), samplerate=44100, channels=1)
        sd.wait()
        write("speech.wav", 44100, (rec * 32767).astype(np.int16))
        st.rerun()
    if b2.button("🛑 ANALYZE"):
        if os.path.exists("speech.wav"):
            res = model_w.transcribe("speech.wav")
            tempo, avg_pitch, y, sr = get_vocal_stats("speech.wav")
            tips, found = get_smart_feedback(st.session_state.mode, res['text'], tempo)
            st.session_state.update({'transcription': res['text'], 'tips': tips, 'filler_count': len(found), 'tempo': tempo, 'avg_pitch': avg_pitch, 'analysis_ready': True, 'conf_score': int(85+(tempo/20)) if tempo < 145 else 60})
            st.rerun()

# --- 6. REPORT & WAVEFORM ---
if st.session_state.get('analysis_ready'):
    t_text = st.session_state.transcription
    for f in ["um", "ah", "basically"]: t_text = t_text.replace(f, f'<span class="highlight">{f}</span>')
    tips_html = "".join([f'<div>• {t}</div>' for t in st.session_state.get('tips', [])])
    
    st.markdown(f'<div style="background:rgba(255,255,255,0.02); border:1px solid #00f2fe; border-radius:20px; padding:30px;"><p style="color:#00f2fe; font-weight:bold;">💡 REPORT</p><p>"{t_text}"</p><hr><div style="display:grid; grid-template-columns: 2fr 1fr; gap:30px;"><div>{tips_html}</div><div class="feature-box">BPM: {st.session_state.tempo:.0f}<br>Pitch: {st.session_state.avg_pitch:.1f} Hz</div></div></div>', unsafe_allow_html=True)
    
    # Pro Waveform Visualization
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.set_facecolor('#08081a')
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#ff7070')
    ax.axis('off')
    st.pyplot(fig)

# --- 7. AI SCRIPT BOT ---
st.write("##")
st.markdown('<p style="color:#00f2fe; font-weight:bold; font-size:22px;">🤖 SCRIPT GENERATOR BOT</p>', unsafe_allow_html=True)
p = st.text_input("Enter topic for a practice script:")
if st.button("Generate AI Script ✨"):
    if p:
        response = ai_model.generate_content(f"Write a 1-minute practice script for {st.session_state.mode}: {p}")
        st.session_state.chat_history.append({"a": response.text})
for chat in reversed(st.session_state.chat_history):
    st.success(chat['a'])