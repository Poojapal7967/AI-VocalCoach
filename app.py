import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- CENTERED NEON UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at center, #1a1a3a 0%, #050510 100%); color: #ffffff; }
    
    /* Centering All Content */
    .main .block-container {
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    /* Metric Centering & Styling */
    [data-testid="stMetricValue"] { 
        color: #ffffff !important; 
        font-size: 50px !important; 
        font-weight: 900 !important;
        display: flex;
        justify-content: center;
    }
    [data-testid="stMetricLabel"] { 
        color: #00f2fe !important; 
        font-size: 18px !important;
        display: flex;
        justify-content: center;
        width: 100%;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 25px !important;
        padding: 30px !important;
        text-align: center;
    }

    /* Improvement Box Centering */
    .improvement-box {
        background: rgba(255, 75, 75, 0.1);
        border: 1px solid #ff4b4b;
        padding: 20px;
        border-radius: 15px;
        margin: 10px auto;
        max-width: 800px;
        text-align: center;
    }

    /* Button Styling */
    .stButton>button {
        width: 60% !important;
        margin: 0 auto;
        display: block;
        background: linear-gradient(90deg, #7000ff, #00f2fe) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 15px !important;
        font-size: 20px !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP UI ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("Master Your Voice with Centered Professional Feedback")
st.markdown("---")

# Settings in Centered Layout
c1, c2, c3 = st.columns([1, 1, 1])
with c1: duration = st.select_slider("Duration (sec)", options=[5, 10, 15], value=5)
with c2: language = st.selectbox("Language", ["English", "Hindi"])
with c3: goal = st.selectbox("Goal", ["Public Speaking", "Anchoring", "Teaching", "Interview"])

st.write("##")
if st.button(f"✨ START {goal.upper()} SESSION"):
    fs = 44100
    with st.spinner("🎤 Analyzing Your Voice..."):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        
        # Volume Normalization & Boost
        if np.max(np.abs(recording)) > 0:
            recording = (recording / np.max(np.abs(recording))) * 2.5
            recording = np.clip(recording, -1, 1)
        write('speech.wav', fs, recording)
    
    st.audio('speech.wav')

    # AI Processing
    result = model.transcribe("speech.wav", language=("en" if language=="English" else "hi"))
    text = result['text']
    y, sr = librosa.load("speech.wav")
    
    # Analysis
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    wpm = int(len(text.split()) / (duration / 60))
    fillers = ["um", "uh", "ah", "like", "hmm", "matlab", "toh"]
    filler_count = sum(1 for word in text.split() if word.lower().strip(",.") in fillers)

    # --- CENTERED RESULTS REPORT ---
    st.markdown(f"### 📊 Live Analysis: {goal}")
    
    # Centered Waveform
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#00f2fe')
    ax.set_axis_off()
    fig.patch.set_facecolor('#050510')
    st.pyplot(fig)

    # Centered Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Speech Pace", f"{wpm} WPM")
    m2.metric("Filler Words", filler_count)
    m3.metric("Voice Energy", "High" if pitch_var > 70 else "Low")

    st.write("##")
    st.info(f"**Transcription:** {text}")

    # Improvement Tracker (Centered)
    st.subheader("🚀 Improvement Insights")
    improvements = []
    if wpm > 160: improvements.append("⚠️ **Too Fast:** Please slow down for better clarity.")
    if filler_count > 2: improvements.append(f"⚠️ **Fillers:** You used {filler_count} filler words. Focus on silent pauses.")
    if pitch_var < 40: improvements.append("⚠️ **Monotone:** Add more vocal variety to keep the audience engaged.")

    if not improvements:
        st.success("🌟 Your delivery was excellent! Keep it up.")
    else:
        for imp in improvements:
            st.markdown(f'<div class="improvement-box">{imp}</div>', unsafe_allow_html=True)