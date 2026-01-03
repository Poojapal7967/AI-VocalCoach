import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import time

# --- ULTIMATE HIGH-CONTRAST UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    /* Dark Premium Background */
    .stApp {
        background: #0e1117;
        color: white;
    }
    
    /* Metrics Styling - Numbers ko White aur bada karne ke liye */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 50px !important;
        font-weight: bold !important;
    }
    
    /* Metric Labels (Confidence, Pace etc.) */
    [data-testid="stMetricLabel"] {
        color: #00f2fe !important;
        font-size: 20px !important;
    }

    /* Card Boxes Styling */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid #1e2130 !important;
        padding: 30px !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #00f2fe, #4facfe) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px !important;
        font-size: 18px !important;
    }

    /* Transcription Expanders */
    .stExpander {
        background: #1e2130 !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# AI Model Loading
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP UI ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("### Analyze your confidence, tone, and speaking speed.")
st.markdown("---")

col_main, col_sidebar = st.columns([2, 1])

with col_sidebar:
    st.write("### ⚙️ Settings")
    duration = st.slider("Recording Duration (sec)", 3, 15, 5)
    st.info("Tip: Higher duration gives better AI insights.")

with col_main:
    if st.button("🔴 Start Live Analysis"):
        fs = 44100
        with st.spinner("✨ Recording..."):
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            write('speech.wav', fs, recording)
        
        st.success("✅ Voice Captured!")
        st.audio('speech.wav')

        # AI PROCESSING
        with st.spinner("AI is evaluating your performance..."):
            result = model.transcribe("speech.wav")
            text = result['text']
            
            y, sr = librosa.load("speech.wav")
            pitches, _ = librosa.piptrack(y=y, sr=sr)
            avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            words = text.split()
            wpm = int(len(words) / (duration / 60))
            fillers = ["um", "uh", "ah", "like", "hmm", "actually"]
            filler_count = sum(1 for word in words if word.lower().strip(",.") in fillers)
            
            # Score logic
            score = max(10, 100 - (filler_count * 15))

        # --- FINAL RESULTS DISPLAY ---
        st.markdown("## 📊 Performance Analytics")
        
        # Ye columns boxes ko bada dikhayenge
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence Score", f"{score}%")
        c2.metric("Filler Words", filler_count)
        c3.metric("Speech Pace", f"{wpm} WPM")

        with st.expander("📝 View Transcription"):
            st.write(f"**You said:** *\"{text}\"*")

        # --- STYLISH TIPS ---
        st.subheader("💡 AI Coaching Insights")
        if score > 80:
            st.success("🌟 Excellent! Your delivery is sharp and authoritative.")
        else:
            st.warning("⚡ Try to reduce filler words like 'um' and 'uh'.")
            
        if wpm < 110:
            st.info("🐢 You are speaking a bit slowly. Increase your pace.")
        elif wpm > 170:
            st.info("🚀 Speaking too fast! Take small pauses.")