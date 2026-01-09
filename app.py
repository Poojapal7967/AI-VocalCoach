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

    div.stAudio { margin-top: 15px; width: 95% !important; border-radius: 12px; }

    .stButton > button {
        width: 100% !important; border-radius: 50px !important;
        padding: 15px 0px !important; font-weight: 800 !important;
        text-transform: uppercase; letter-spacing: 2px; transition: 0.3s;
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
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo, avg_pitch, y, sr

def get_smart_feedback(transcript, tempo):
    fillers = ["um", "ah", "basically", "actually", "like", "hmmm"]
    words = transcript.lower().split()
    found_fillers = [w for w in words if w in fillers]
    tips = []
    if tempo > 140: tips.append("⚠️ **Speed:** Too fast! Try to slow down for better clarity.")
    elif tempo < 80: tips.append("⚠️ **Speed:** A bit slow. Try to be more energetic.")
    else: tips.append("✅ **Speed:** Perfect pacing! Maintain this rhythm.")
    
    if len(found_fillers) > 0:
        tips.append(f"⚠️ **Clarity:** Found {len(found_fillers)} filler words. Practice pausing.")
    else:
        tips.append("✅ **Clarity:** Great job! No major filler words detected.")
    return tips, found_fillers

# --- 3. LOGIC INITIALIZATION ---
if 'recording' not in st.session_state: st.session_state.recording = False
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False

@st.cache_resource
def load_model(): return whisper.load_model("base")
model = load_model()

# --- 4. PERFORMANCE GRID ---
st.write("##")
st.markdown('<p style="color:#ffcc00; font-weight:800; font-size:20px; letter-spacing:1px;">📊 PERFORMANCE REPORT</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    # Naya Modern 100 Scale Card (image_e41ca4 Style)
    score = st.session_state.get('conf_score', 85)
    sentiment_text = "POSITIVE" if score >= 70 else "NEUTRAL"
    sentiment_emoji = "😊" if score >= 70 else "😐"
    sentiment_color = "#4caf50" if score >= 70 else "#ff9800"

    st.markdown(f"""
        <div class="neon-card glow-red">
            <div class="label-sub">CONFIDENCE SCORE</div>
            <div class="big-num">{score}<span style="font-size:30px;">/100</span></div>
            <div style="height:8px; background:rgba(255,255,255,0.1); border-radius:10px; width:85%; margin:15px auto;">
                <div style="height:100%; width:{score}%; background:linear-gradient(to right, #00f2fe, #4caf50); border-radius:10px;"></div>
            </div>
            <div style="display:inline-flex; align-items:center; background:rgba(255,255,255,0.05); padding:5px 15px; border-radius:50px; border:1px solid {sentiment_color};">
                <span style="font-size:18px; margin-right:8px;">{sentiment_emoji}</span>
                <span style="color:{sentiment_color}; font-weight:bold; font-size:12px; letter-spacing:1px;">SENTIMENT: {sentiment_text}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown('<div class="neon-card glow-orange">', unsafe_allow_html=True)
    if st.session_state.recording:
        st.markdown('<div><span class="recording-dot"></span><span style="color:#ff4b4b; font-weight:bold;">LIVE RECORDING</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:45px; margin-bottom:10px;">🎙️</div>', unsafe_allow_html=True)
    
    if os.path.exists("speech.wav"):
        st.audio("speech.wav")
        st.markdown('<div class="label-sub">VOICE READY</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="label-sub">NO DATA FOUND</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    filler_p = st.session_state.get('filler_count', 3)
    st.markdown(f'<div class="neon-card glow-cyan"><div class="big-num">{filler_p}%</div><div class="label-sub">Fillers Found</div></div>', unsafe_allow_html=True)

# --- 5. CONTROLS & ANALYSIS ---
st.write("##")
_, c_btn, _ = st.columns([1, 2, 1])

with c_btn:
    b1, b2 = st.columns(2)
    with b1:
        if st.button("🎤 START"):
            st.session_state.recording = True
            st.session_state.analysis_ready = False
            fs, duration = 44100, 5
            rec_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            write("speech.wav", fs, (rec_data * 32767).astype(np.int16))
            st.session_state.recording = False
            st.session_state.analysis_ready = True
            st.rerun()

    with b2:
        if st.button("🛑 ANALYZE"):
            if os.path.exists("speech.wav"):
                with st.spinner("AI Analysis..."):
                    result = model.transcribe("speech.wav")
                    st.session_state.transcription = result['text']
                    tempo, _, _, _ = get_vocal_stats("speech.wav")
                    tips, fillers = get_smart_feedback(result['text'], tempo)
                    st.session_state.tips = tips
                    st.session_state.filler_count = len(fillers)
                    st.session_state.conf_score = int(80 + (tempo/10)) if tempo < 140 else 65
                    st.session_state.analysis_ready = True
                    st.rerun()

# --- 6. ADVANCED DASHBOARD ---
if st.session_state.analysis_ready and 'transcription' in st.session_state:
    st.write("##")
    t_text = st.session_state.transcription
    for f in ["um", "ah", "basically", "actually", "like"]:
        t_text = t_text.replace(f, f'<span class="highlight">{f}</span>')
    
    st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); border: 1.5px solid #00f2fe; border-radius: 20px; padding: 35px; box-shadow: 0 0 20px rgba(0, 242, 254, 0.2);">
            <p style="color:#00f2fe; font-weight:900; font-size:20px; text-transform:uppercase;">💡 AI Transcript & Feedback</p>
            <p style="color:#fff; font-size:18px; font-style:italic;">"{t_text}"</p>
            <hr style="border-color:rgba(255,255,255,0.1);">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                <div>
                    <p style="color:#ffcc00; font-weight:bold;">Coaching Tips:</p>
                    {"".join([f'<div style="margin-bottom:5px;">{tip}</div>' for tip in st.session_state.get('tips', [])])}
                </div>
                <div style="text-align:center;">
                    <p style="color:#ffcc00; font-weight:bold;">Speech Waveform</p>
                    <div id="plot_area"></div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    tempo, pitch, y, sr = get_vocal_stats("speech.wav")
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_facecolor('#08081a')
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#ff4b4b', alpha=0.7)
    ax.axis('off')
    st.pyplot(fig)