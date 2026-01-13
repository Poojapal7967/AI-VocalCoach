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

# --- 1. PREMIUM NEON UI (NO REDUCTIONS) ---
st.set_page_config(page_title="AI Vocal Coach Pro", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #08081a; color: #ffffff; font-family: 'Inter', sans-serif; }
    .neon-card { background: rgba(255, 255, 255, 0.04); border-radius: 25px; padding: 40px; text-align: center; height: 350px; border: 1px solid rgba(255, 255, 255, 0.1); }
    .big-num { font-size: 80px; font-weight: 900; line-height: 1; margin-bottom: 10px; }
    .label-sub { font-size: 14px; color: #a0a0b0; text-transform: uppercase; letter-spacing: 2px; }
    .feature-box { background: rgba(0, 242, 254, 0.05); border-radius: 15px; padding: 15px; border-left: 4px solid #00f2fe; margin-top: 10px; text-align: left; }
    .improvement-box { background: rgba(0, 255, 127, 0.05); border: 1px solid #00ff7f; border-radius: 20px; padding: 25px; margin-top: 20px; }
    .highlight { color: #ff4b4b; font-weight: bold; text-decoration: underline; }
    div.stAudio { margin-top: 15px; border-radius: 50px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. BACKEND LOGIC ---
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

# --- 3. SESSION INITIALIZATION ---
if 'mode' not in st.session_state: st.session_state.mode = "🎤 Public Speaking"
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'vocal_history' not in st.session_state: st.session_state.vocal_history = [] 
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False

@st.cache_resource
def load_whisper(): return whisper.load_model("base")
model_w = load_whisper()

# --- 4. DASHBOARD UI ---
st.markdown('<p style="color:#00f2fe; font-weight:800; font-size:24px;">🎯 MODE SELECTOR</p>', unsafe_allow_html=True)
st.session_state.mode = st.radio("", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"], horizontal=True)

st.write("##")
c1, c2, c3 = st.columns(3)

with c1: 
    score = st.session_state.get('conf_score', 85)
    st.markdown(f'<div class="neon-card"><div class="label-sub">CONFIDENCE SCORE</div><div class="big-num">{score}</div><div style="height:8px; background:rgba(255,255,255,0.1); border-radius:10px; width:80%; margin:20px auto;"><div style="height:100%; width:{score}%; background:linear-gradient(to right, #00f2fe, #4caf50); border-radius:10px;"></div></div><div style="color:#4caf50; font-size:12px;">😊 SENTIMENT: POSITIVE</div></div>', unsafe_allow_html=True)

with c2: 
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    if os.path.exists("speech.wav"):
        st.markdown('<div style="font-size:40px">🎙️</div>', unsafe_allow_html=True)
        st.audio("speech.wav")
        st.markdown(f'<div class="label-sub">{st.session_state.mode} READY</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="label-sub">NO DATA FOUND</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3: 
    st.markdown(f'<div class="neon-card"><div class="big-num">{st.session_state.get("filler_count", 0)}%</div><div class="label-sub">Fillers Found</div></div>', unsafe_allow_html=True)

# --- 5. STABLE CONTROLS ---
st.write("##")
_, cb, _ = st.columns([1, 2, 1])
with cb:
    b1, b2 = st.columns(2)
    if b1.button("🎤 START RECORDING"):
        with st.status("Recording... Speak Now!"):
            fs = 44100
            duration = 5
            # Precise float32 recording to fix sounddevice issues
            rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            write("speech.wav", fs, (rec * 32767).astype(np.int16))
            st.rerun()
    if b2.button("🛑 ANALYZE"):
        if os.path.exists("speech.wav"):
            with st.spinner("Analyzing..."):
                res = model_w.transcribe("speech.wav")
                tempo, avg_pitch, y, sr = get_vocal_stats("speech.wav")
                tips, found = get_smart_feedback(st.session_state.mode, res['text'], tempo)
                f_perc = int((len(found) / len(res['text'].split())) * 100) if len(res['text'].split()) > 0 else 0
                score_now = int(85+(tempo/20)) if tempo < 145 else 60
                
                # Comparative History Save
                st.session_state.vocal_history.append({'score': score_now, 'fillers': f_perc, 'tempo': tempo})
                st.session_state.update({
                    'transcription': res['text'], 'tips': tips, 
                    'filler_count': f_perc, 'tempo': tempo, 
                    'avg_pitch': avg_pitch, 'analysis_ready': True, 
                    'conf_score': score_now
                })
                st.rerun()

# --- 6. REPORT & COMPARATIVE ANALYSIS ---
if st.session_state.get('analysis_ready'):
    if len(st.session_state.vocal_history) > 1:
        prev, now = st.session_state.vocal_history[-2], st.session_state.vocal_history[-1]
        
        st.markdown('<div class="improvement-box">', unsafe_allow_html=True)
        st.markdown('<p style="color:#00ff7f; font-weight:bold; font-size:20px;">📈 COMPARATIVE ANALYSIS (Improvement Report)</p>', unsafe_allow_html=True)
        ca, cb = st.columns([1, 2])
        with ca:
            diff = now['score'] - prev['score']
            color = "#00ff7f" if diff >= 0 else "#ff4b4b"
            st.markdown(f"**Confidence:** <span style='color:{color}'>{'+' if diff >=0 else ''}{diff} pts</span>", unsafe_allow_html=True)
            f_diff = prev['fillers'] - now['fillers']
            f_color = "#00ff7f" if f_diff >= 0 else "#ff4b4b"
            st.markdown(f"**Clarity:** <span style='color:{f_color}'>{'+' if f_diff >=0 else ''}{f_diff}% improved</span>", unsafe_allow_html=True)
        
        with cb:
            # Comparison Bar Chart
            fig2, ax2 = plt.subplots(figsize=(6, 2.5))
            ax2.set_facecolor('#08081a')
            labels = ['Score', 'Fillers']
            prev_vals = [prev['score'], prev['fillers']]
            now_vals = [now['score'], now['fillers']]
            x = np.arange(len(labels))
            ax2.bar(x - 0.2, prev_vals, 0.4, label='Last', color='#444466')
            ax2.bar(x + 0.2, now_vals, 0.4, label='Current', color='#00f2fe')
            ax2.set_xticks(x); ax2.set_xticklabels(labels, color='white')
            ax2.legend(); plt.yticks(color='white')
            st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    # Vocal Report & Waveform
    t_text = st.session_state.transcription
    for f in ["um", "ah", "basically"]: t_text = t_text.replace(f, f'<span class="highlight">{f}</span>')
    tips_joined = "".join([f'<div>• {t}</div>' for t in st.session_state.get('tips', [])])
    
    st.markdown(f'<div style="background:rgba(255,255,255,0.02); border:1px solid #00f2fe; border-radius:20px; padding:30px;"><p style="color:#00f2fe; font-weight:bold;">💡 VOCAL REPORT</p><p>"{t_text}"</p><hr><div style="display:grid; grid-template-columns: 2fr 1fr; gap:30px;"><div>{tips_joined}</div><div class="feature-box">BPM: {st.session_state.tempo:.0f}<br>Pitch: {st.session_state.avg_pitch:.1f} Hz</div></div></div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(15, 2))
    ax.set_facecolor('#08081a')
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#ff7070')
    ax.axis('off')
    st.pyplot(fig)

# --- 7. AI SCRIPT BOT ---
st.write("##")
st.markdown('<p style="color:#00f2fe; font-weight:bold; font-size:22px;">🤖 SCRIPT GENERATOR</p>', unsafe_allow_html=True)
p = st.text_input("Enter topic for practice script:")
if st.button("Generate AI Script ✨"):
    if p:
        response = ai_model.generate_content(f"Write a script for {st.session_state.mode}: {p}")
        st.session_state.chat_history.append({"a": response.text})
for chat in reversed(st.session_state.chat_history):
    st.success(chat['a'])