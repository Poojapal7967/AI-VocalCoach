import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import queue
import sys

# --- UI STYLING (Aapka Pehle Wala) ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at center, #1a1a3a 0%, #050510 100%); color: #ffffff; }
    .main .block-container { text-align: center; display: flex; flex-direction: column; align-items: center; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 50px !important; font-weight: 900 !important; display: flex; justify-content: center; }
    [data-testid="stMetricLabel"] { color: #00f2fe !important; font-size: 18px !important; display: flex; justify-content: center; width: 100%; }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.08) !important; backdrop-filter: blur(20px) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 25px !important; padding: 30px !important; text-align: center; }
    .improvement-box { background: rgba(255, 75, 75, 0.1); border: 1px solid #ff4b4b; padding: 20px; border-radius: 15px; margin: 10px auto; max-width: 800px; text-align: center; }
    
    /* Dynamic Button Color */
    .stButton>button {
        width: 60% !important; margin: 0 auto; display: block;
        color: white !important; border-radius: 50px !important;
        padding: 15px !important; font-size: 20px !important; font-weight: bold !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- RECORDING LOGIC ---
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []

def callback(indata, frames, time, status):
    """Indata ko buffer mein add karne ke liye."""
    if st.session_state.recording:
        st.session_state.audio_buffer.append(indata.copy())

# Model Load
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP UI ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("Ab manual Start/Stop ke saath apni speech master karein")
st.markdown("---")

c1, c2 = st.columns(2)
with c1: language = st.selectbox("Language", ["English", "Hindi"])
with c2: goal = st.selectbox("Goal", ["Public Speaking", "Anchoring", "Teaching", "Interview"])

st.write("##")

# --- MANUAL CONTROL BUTTONS ---
btn_col1, btn_col2, btn_col3 = st.columns([1,2,1])

with btn_col2:
    if not st.session_state.recording:
        if st.button("🎤 START RECORDING", type="primary", use_container_width=True):
            st.session_state.recording = True
            st.session_state.audio_buffer = []
            st.rerun()
    else:
        if st.button("🛑 STOP & ANALYZE", type="secondary", use_container_width=True):
            st.session_state.recording = False
            
            # Combine buffer and save
            if st.session_state.audio_buffer:
                audio_fp32 = np.concatenate(st.session_state.audio_buffer, axis=0)
                # Normalization
                if np.max(np.abs(audio_fp32)) > 0:
                    audio_fp32 = (audio_fp32 / np.max(np.abs(audio_fp32))) * 0.9
                write('speech.wav', 44100, audio_fp32)
                st.session_state.process_done = True
            st.rerun()

# --- BACKGROUND STREAMING ---
# Ye part background mein mic open rakhta hai
if st.session_state.recording:
    st.warning("🔴 Recording in Progress... Bolte rahiye!")
    # InputStream setup (Simple version for manual stop)
    # Note: Professional manual stop ke liye webrtc best hota hai, 
    # but Streamlit session state se hum ye background mein handle kar sakte hain.
    with sd.InputStream(samplerate=44100, channels=1, callback=callback):
        # Jab tak recording True hai, ye block karega
        while st.session_state.recording:
            import time
            time.sleep(0.1)

# --- ANALYSIS REPORT (Same as yours) ---
if 'process_done' in st.session_state and st.session_state.process_done:
    del st.session_state.process_done
    
    with st.spinner("⌛ AI Analyzing..."):
        st.audio('speech.wav')
        result = model.transcribe("speech.wav", language=("en" if language=="English" else "hi"))
        text = result['text']
        y, sr = librosa.load("speech.wav")
        
        # Calculations
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        duration_actual = len(y) / sr
        wpm = int(len(text.split()) / (duration_actual / 60)) if duration_actual > 0 else 0
        fillers = ["um", "uh", "ah", "like", "hmm", "matlab", "toh"]
        filler_count = sum(1 for word in text.split() if word.lower().strip(",.") in fillers)

        st.markdown(f"### 📊 Live Analysis: {goal}")
        
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#00f2fe')
        ax.set_axis_off()
        fig.patch.set_facecolor('#050510')
        st.pyplot(fig)

        m1, m2, m3 = st.columns(3)
        m1.metric("Speech Pace", f"{wpm} WPM")
        m2.metric("Filler Words", filler_count)
        m3.metric("Voice Energy", "High" if pitch_var > 70 else "Low")

        st.info(f"**Transcription:** {text}")

        # Improvements
        improvements = []
        if wpm > 160: improvements.append("⚠️ **Too Fast:** Please slow down.")
        if filler_count > 2: improvements.append(f"⚠️ **Fillers:** Used {filler_count} fillers.")
        if pitch_var < 40: improvements.append("⚠️ **Monotone:** Add vocal variety.")

        if not improvements:
            st.success("🌟 Excellent delivery!")
        else:
            for imp in improvements:
                st.markdown(f'<div class="improvement-box">{imp}</div>', unsafe_allow_html=True)