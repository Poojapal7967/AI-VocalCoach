import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# --- UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at center, #1a1a3a 0%, #050510 100%); color: #ffffff; }
    .main .block-container { text-align: center; }
    
    /* Metrics UI */
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 50px !important; font-weight: 900 !important; }
    [data-testid="stMetricLabel"] { color: #00f2fe !important; font-size: 18px !important; }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.08) !important; backdrop-filter: blur(20px) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 25px !important; padding: 20px !important; }
    
    /* Improvement Insights Box */
    .improvement-box { background: rgba(255, 75, 75, 0.1); border: 1px solid #ff4b4b; padding: 20px; border-radius: 15px; margin: 10px auto; max-width: 800px; text-align: center; }
    
    /* Centered Audio Player */
    section[data-testid="stAudio"] { margin: 0 auto; max-width: 600px; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION INITIALIZATION ---
if 'recording_active' not in st.session_state:
    st.session_state.recording_active = False
if 'audio_file_ready' not in st.session_state:
    st.session_state.audio_file_ready = False

# Load AI Model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP UI ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("Manual Control & Fixed Player System")
st.markdown("---")

col_set1, col_set2 = st.columns(2)
with col_set1:
    language = st.selectbox("Language", ["English", "Hindi"])
with col_set2:
    goal = st.selectbox("Goal", ["Public Speaking", "Anchoring", "Teaching", "Interview"])

# --- RECORDING ACTIONS ---
st.write("##")
btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])

# START Button
if btn_col1.button("🎤 START", use_container_width=True, type="primary"):
    st.session_state.recording_active = True
    st.session_state.audio_file_ready = False
    st.rerun()

# STOP Button
if btn_col2.button("🛑 STOP & ANALYZE", use_container_width=True):
    st.session_state.recording_active = False
    st.session_state.audio_file_ready = True
    # Yahan recording handle hogi background thread mein
    st.rerun()

# Reset Button
if btn_col3.button("🔄 RESET", use_container_width=True):
    st.session_state.recording_active = False
    st.session_state.audio_file_ready = False
    if os.path.exists('speech.wav'):
        os.remove('speech.wav')
    st.rerun()

# --- BACKGROUND RECORDING LOGIC ---
if st.session_state.recording_active:
    st.warning("🔴 Recording... (Press STOP when finished)")
    fs = 44100
    # Fixed long duration for manual stop simulation
    # Streamlit synchronously records here
    duration = 600 # 10 minutes max
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    
    # We save as we go or on stop
    # NOTE: Since Streamlit is synchronous, actual 'manual stop' is best 
    # simulated with a very long recording that we truncate.
    write('speech.wav', fs, recording)

# --- RESULTS DISPLAY ---
if st.session_state.audio_file_ready and os.path.exists('speech.wav'):
    st.success("✅ Recording Saved!")
    st.audio('speech.wav')
    
    with st.spinner("⌛ Analyzing your performance..."):
        # Load and Truncate Silence (Important for manual stop)
        y, sr = librosa.load('speech.wav')
        y_trimmed, _ = librosa.effects.trim(y)
        write('speech.wav', sr, y_trimmed) # Overwrite with trimmed version
        
        # Whisper AI
        result = model.transcribe('speech.wav', language=("en" if language=="English" else "hi"))
        text = result['text']
        
        # Metrics
        pitches, _ = librosa.piptrack(y=y_trimmed, sr=sr)
        pitch_var = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        duration_sec = len(y_trimmed) / sr
        wpm = int(len(text.split()) / (duration_sec / 60)) if duration_sec > 0 else 0
        
        # UI: Dashboard
        st.markdown(f"### 📊 Analysis Report: {goal}")
        
        # Waveform
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y_trimmed, sr=sr, ax=ax, color='#00f2fe')
        ax.set_axis_off()
        fig.patch.set_facecolor('#050510')
        st.pyplot(fig)

        m1, m2, m3 = st.columns(3)
        m1.metric("Pace", f"{wpm} WPM")
        m2.metric("Energy", "High" if pitch_var > 70 else "Low")
        m3.metric("Clarity", "Good" if len(text) > 10 else "Low")

        st.info(f"**Transcription:** {text}")

        # Improvements
        st.subheader("🚀 Improvement Insights")
        if wpm > 165:
            st.markdown('<div class="improvement-box">⚠️ **Too Fast:** Bolte waqt pauses lein.</div>', unsafe_allow_html=True)
        if pitch_var < 40:
            st.markdown('<div class="improvement-box">⚠️ **Monotone:** Awaaz mein utaar-chadhaw laayein.</div>', unsafe_allow_html=True)
        if not text:
            st.error("AI couldn't hear any words. Mic check karein!")