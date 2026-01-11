import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# --- 1. ULTRA-NEON ATTRATIVE UI ---
st.set_page_config(page_title="AI Vocal Coach Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #08081a; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    @keyframes pulse-red {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 12px rgba(255, 75, 75, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    .recording-dot {
        width: 14px; height: 14px; background: #ff4b4b; border-radius: 50%;
        display: inline-block; margin-right: 10px; animation: pulse-red 1.5s infinite;
    }

    .neon-card {
        background: rgba(255, 255, 255, 0.04); border-radius: 25px; padding: 45px 20px;
        text-align: center; backdrop-filter: blur(20px); height: 350px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.1); transition: 0.5s;
    }
    .glow-red { border-color: #ff4b4b !important; box-shadow: 0 0 30px rgba(255, 75, 75, 0.5); }
    .glow-orange { border-color: #ff9800 !important; box-shadow: 0 0 30px rgba(255, 152, 0, 0.5); }
    .glow-cyan { border-color: #00f2fe !important; box-shadow: 0 0 30px rgba(0, 242, 254, 0.5); }

    .big-num { font-size: 85px; font-weight: 900; line-height: 1; margin-bottom: 10px; }
    .label-sub { font-size: 14px; color: #a0a0b0; text-transform: uppercase; letter-spacing: 3px; }

    .feature-box {
        background: rgba(0, 242, 254, 0.05); border-radius: 15px;
        padding: 15px; border-left: 4px solid #00f2fe; margin-top: 10px; text-align: left;
    }

    div.stAudio { margin-top: 15px; width: 95% !important; border-radius: 12px; }

    .stButton > button {
        width: 100% !important; border-radius: 50px !important;
        padding: 15px 0px !important; font-weight: 800 !important;
        text-transform: uppercase; letter-spacing: 2px;
    }
    div[data-testid="column"]:nth-of-type(1) .stButton > button {
        border: 2.5px solid #ff4b4b !important; box-shadow: 0 0 25px rgba(255, 75, 75, 0.6);
        background: transparent !important; color: white !important;
    }
    div[data-testid="column"]:nth-of-type(2) .stButton > button {
        border: 2.5px solid #ffffff !important; box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
        background: transparent !important; color: white !important;
    }
    
    .highlight { color: #ff4b4b; font-weight: bold; text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ADVANCED ANALYSIS FUNCTIONS ---
def get_vocal_stats(file_path):
    y, sr = librosa.load(file_path)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    # FIX: Explicit float conversion to avoid NumPy ndarray formatting error
    avg_pitch = float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # Convert tempo to float if it is an array
    if isinstance(tempo, (list, np.ndarray)): tempo = tempo[0]
    return float(tempo), avg_pitch, y, sr

def get_smart_feedback(mode, transcript, tempo):
    fillers = ["um", "ah", "basically", "actually", "like", "hmmm"]
    words = transcript.lower().split()
    found_fillers = [w for w in words if w in fillers]
    tips = []
    
    # Base Clarity
    if len(found_fillers) > 0:
        tips.append(f"⚠️ **Clarity:** Found {len(found_fillers)} fillers. Practice clean pauses.")
    else:
        tips.append("✅ **Clarity:** Excellent! No major filler words detected.")

    # Multi-Mode Logic
    if mode == "💼 Interview":
        tips.append("🤝 **Directness:** Keep answers concise. Authority comes from clear finishes.")
    elif mode == "👨‍🏫 Teaching":
        tips.append("📖 **Articulation:** Speak slow on complex terms for student retention.")
    elif mode == "🎤 Public Speaking":
        tips.append("🌟 **Projection:** Vary your volume levels to keep high audience engagement.")
    else: # Anchoring
        tips.append("✨ **Energy:** Maintain a rhythmic flow. Consistency is your best tool.")
    
    return tips, found_fillers

# --- 3. LOGIC INITIALIZATION ---
if 'recording' not in st.session_state: st.session_state.recording = False
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
if 'mode' not in st.session_state: st.session_state.mode = "🎤 Public Speaking"

@st.cache_resource
def load_model(): return whisper.load_model("base")
model = load_model()

# --- 4. MODE NAVIGATION ---
st.markdown('<p style="color:#00f2fe; font-weight:800; font-size:24px;">🎯 TRAINING MODE SELECTOR</p>', unsafe_allow_html=True)
st.session_state.mode = st.radio("", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"], horizontal=True)

# --- 5. PERFORMANCE GRID ---
st.write("##")
st.markdown('<p style="color:#ffcc00; font-weight:800; font-size:20px;">📊 PERFORMANCE REPORT</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1: # Confidence Card
    score = st.session_state.get('conf_score', 85)
    st.markdown(f"""
        <div class="neon-card glow-red">
            <div class="label-sub">CONFIDENCE SCORE</div>
            <div class="big-num">{score}<span style="font-size:30px;">/100</span></div>
            <div style="height:8px; background:rgba(255,255,255,0.1); border-radius:10px; width:85%; margin:15px auto;">
                <div style="height:100%; width:{score}%; background:linear-gradient(to right, #00f2fe, #4caf50); border-radius:10px;"></div>
            </div>
            <div style="display:inline-flex; align-items:center; background:rgba(255,255,255,0.05); padding:5px 15px; border-radius:50px; border:1px solid #4caf50; color:#4caf50; font-weight:bold; font-size:12px;">
                😊 SENTIMENT: POSITIVE
            </div>
        </div>
    """, unsafe_allow_html=True)

with c2: # Center Card
    st.markdown('<div class="neon-card glow-orange">', unsafe_allow_html=True)
    if st.session_state.recording:
        st.markdown('<div><span class="recording-dot"></span><span style="color:#ff4b4b; font-weight:bold;">LIVE RECORDING</span></div>', unsafe_allow_html=True)
    else:
        if os.path.exists("speech.wav"):
            st.markdown('<div style="font-size:45px; margin-bottom:10px;">🎙️</div>', unsafe_allow_html=True)
            st.audio("speech.wav")
            st.markdown(f'<div class="label-sub">{st.session_state.mode} READY</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="label-sub">NO DATA FOUND</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3: # Filler Card
    filler_p = st.session_state.get('filler_count', 3)
    st.markdown(f'<div class="neon-card glow-cyan"><div class="big-num">{filler_p}%</div><div class="label-sub">Fillers Found</div></div>', unsafe_allow_html=True)

# --- 6. CONTROLS ---
st.write("##")
_, c_btn, _ = st.columns([1, 2, 1])
with c_btn:
    b1, b2 = st.columns(2)
    if b1.button("🎤 START"):
        st.session_state.recording = True
        fs, duration = 44100, 5
        rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write("speech.wav", fs, (rec * 32767).astype(np.int16))
        st.session_state.recording = False
        st.rerun()
    if b2.button("🛑 ANALYZE"):
        if os.path.exists("speech.wav"):
            with st.spinner("AI Analysis..."):
                res = model.transcribe("speech.wav")
                tempo, avg_pitch, y, sr = get_vocal_stats("speech.wav")
                tips, fillers = get_smart_feedback(st.session_state.mode, res['text'], tempo)
                st.session_state.update({
                    'transcription': res['text'], 'tips': tips, 
                    'filler_count': len(fillers), 'tempo': float(tempo), 
                    'avg_pitch': float(avg_pitch), 'analysis_ready': True,
                    'conf_score': int(85 + (tempo/20)) if tempo < 140 else 60
                })
                st.rerun()

# --- 7. ADVANCED DASHBOARD (Visuals & Health Metrics) ---
if st.session_state.get('analysis_ready'):
    st.write("##")
    t_text = st.session_state.transcription
    for f in ["um", "ah", "basically", "actually", "like"]:
        t_text = t_text.replace(f, f'<span class="highlight">{f}</span>')
    
    # FIX: Proper join for tips HTML to prevent list rendering error
    tips_joined = "".join([f'<div style="margin-bottom:8px; font-size:15px;">• {tip}</div>' for tip in st.session_state.get('tips', [])])
    
    st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); border: 1.5px solid #00f2fe; border-radius: 20px; padding: 35px; box-shadow: 0 0 20px rgba(0, 242, 254, 0.1);">
            <p style="color:#00f2fe; font-weight:900; font-size:20px; text-transform:uppercase;">💡 {st.session_state.mode} COACHING REPORT</p>
            <p style="color:#fff; font-size:18px; font-style:italic;">"{t_text}"</p>
            <hr style="border-color:rgba(255,255,255,0.1); margin: 25px 0;">
            <div style="display:grid; grid-template-columns: 2fr 1fr; gap:30px;">
                <div><p style="color:#ffcc00; font-weight:bold; font-size:18px;">🎯 ACTIONABLE IMPROVEMENT STEPS</p>{tips_joined}</div>
                <div>
                    <p style="color:#ffcc00; font-weight:bold; font-size:18px;">🏥 VOCAL HEALTH METRICS</p>
                    <div class="feature-box">
                        <div style="font-size:10px; color:#888;">SPEED</div>
                        <div style="font-size:18px; font-weight:bold;">{st.session_state.get('tempo', 0):.0f} BPM</div>
                    </div>
                    <div class="feature-box">
                        <div style="font-size:10px; color:#888;">AVG PITCH</div>
                        <div style="font-size:18px; font-weight:bold;">{st.session_state.get('avg_pitch', 0):.1f} Hz</div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Wide Waveform
    tempo, pitch, y, sr = get_vocal_stats("speech.wav")
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.set_facecolor('#08081a')
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#ff7070', alpha=0.9)
    ax.axis('off')
    st.pyplot(fig)