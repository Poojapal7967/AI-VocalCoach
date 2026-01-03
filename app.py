import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at center, #1a1a3a 0%, #050510 100%); color: #ffffff; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 45px !important; font-weight: 900 !important; }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
    }
    .stButton>button {
        width: 100% !important;
        background: linear-gradient(90deg, #7000ff, #00f2fe) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 15px !important;
    }
    .improvement-box {
        background: rgba(255, 75, 75, 0.1);
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP UI ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("<p style='text-align: center;'>Get Real-Time Feedback on Where to Improve</p>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: duration = st.select_slider("Duration (sec)", options=[5, 10, 15], value=5)
with c2: language = st.selectbox("Language", ["English", "Hindi"])
with c3: goal = st.selectbox("Goal", ["Public Speaking", "Anchoring", "Teaching", "Interview"])

if st.button(f"✨ ANALYZE MY {goal.upper()} PERFORMANCE"):
    fs = 44100
    with st.spinner("🎤 Listening & Analyzing..."):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        
        # Volume Boost
        if np.max(np.abs(recording)) > 0:
            recording = (recording / np.max(np.abs(recording))) * 2.0
            recording = np.clip(recording, -1, 1)
        
        write('speech.wav', fs, recording)
    
    # AI Brain
    result = model.transcribe("speech.wav", language=("en" if language=="English" else "hi"))
    text = result['text']
    y, sr = librosa.load("speech.wav")
    
    # Analysis Metrics
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    words = text.split()
    wpm = int(len(words) / (duration / 60))
    fillers = ["um", "uh", "ah", "like", "hmm", "matlab", "toh"]
    filler_count = sum(1 for word in words if word.lower().strip(",.") in fillers)

    # --- SHOW PERFORMANCE REPORT ---
    st.markdown("---")
    st.subheader(f"📊 Live Analysis Report: {goal}")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Speech Pace", f"{wpm} WPM")
    m2.metric("Filler Words", filler_count)
    m3.metric("Voice Energy", "High" if pitch_var > 70 else "Low")

    # --- IMPROVEMENT TRACKER SECTION ---
    st.write("##")
    st.subheader("🚀 Where you need to improve:")
    
    improvements = []
    
    # 1. Pace Check
    if wpm > 160:
        improvements.append("⚠️ **Too Fast:** Aap bahut tez bol rahe hain. Audience ko samajhne ke liye thoda pause lein.")
    elif wpm < 100:
        improvements.append("⚠️ **Too Slow:** Aap thoda dhire bol rahe hain. Thodi energy badhaiye.")
    
    # 2. Filler Check
    if filler_count > 2:
        improvements.append(f"⚠️ **Filler Words:** Aapne {filler_count} baar 'um/uh/matlab' use kiya. Inhe kam karne ke liye practice karein.")
    
    # 3. Tone/Pitch Check
    if pitch_var < 40:
        improvements.append("⚠️ **Monotone Voice:** Aapki awaaz ek jaisi (flat) lag rahi hai. Emotions aur pitch mein utaar-chadhaw laiye.")

    # Display Improvements
    if not improvements:
        st.success("🌟 **Great Job!** Aapki speech mein koi major improvement ki zaroorat nahi hai. Carry on!")
    else:
        for imp in improvements:
            st.markdown(f'<div class="improvement-box">{imp}</div>', unsafe_allow_html=True)

    st.write("##")
    with st.expander("📝 View My Transcription"):
        st.write(text)