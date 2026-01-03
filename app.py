import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import time

# --- ADVANCED UI STYLING (Glassmorphism & Neon) ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1e2130, #0e1117);
        color: #e0e0e0;
    }
    
    /* Neon Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-left: 5px solid #00f2fe; /* Neon Cyan Border */
    }

    /* Professional Button */
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.6);
    }

    /* Headers */
    h1 { color: #ffffff; text-shadow: 0 0 10px rgba(255,255,255,0.2); }
    
    /* Coaching Tips Section */
    .tips-container {
        background: rgba(0, 242, 254, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px dashed #00f2fe;
    }
    </style>
    """, unsafe_allow_html=True)

# --- AI LOGIC ---
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- HEADER SECTION ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("### *Elevate your speaking style with high-fidelity AI analysis.*")
st.markdown("---")

# --- SIDEBAR / CONTROLS ---
col_main, col_sidebar = st.columns([2, 1])

with col_sidebar:
    st.write("### ⚙️ Settings")
    duration = st.slider("Recording Duration (sec)", 3, 15, 5)
    st.info("Tip: Higher duration allows for better analysis of your pace.")

with col_main:
    if st.button("🔴 Start Live Analysis"):
        fs = 44100
        with st.spinner("✨ Listening... Speak with confidence!"):
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            write('speech.wav', fs, recording)
            time.sleep(1) # Visual feel
        
        st.success("✅ Voice Captured!")
        st.audio('speech.wav')

        # --- PROGRESS BAR VISUAL ---
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        # AI PROCESSING
        result = model.transcribe("speech.wav")
        text = result['text']
        y, sr = librosa.load("speech.wav")
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        words = text.split()
        wpm = int(len(words) / (duration / 60))
        fillers = ["um", "uh", "ah", "like", "hmm", "actually"]
        filler_count = sum(1 for word in words if word.lower().strip(",.") in fillers)
        
        # Confidence Score
        score = 100 - (filler_count * 12)
        if wpm < 110 or wpm > 170: score -= 15
        score = max(5, score)

        # --- REPORT UI ---
        st.markdown("## 📊 Performance Analytics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence Level", f"{score}%")
        c2.metric("Filler Words", filler_count)
        c3.metric("Speech Pace", f"{wpm} WPM")

        with st.expander("📝 View Transcription"):
            st.write(f"*{text}*")

        # --- STYLISH COACHING TIPS ---
        st.markdown('<div class="tips-container">', unsafe_allow_html=True)
        st.subheader("💡 AI Coaching Insights")
        
        if score > 80:
            st.success("🌟 **Professional:** Your delivery is sharp and authoritative.")
        elif score > 50:
            st.warning("⚡ **Growth Opportunity:** Good flow, but focus on eliminating fillers.")
        else:
            st.error("📉 **Focus Needed:** Your speech needs more structure and clarity.")

        if wpm < 110:
            st.info("🐢 **Pace:** You are speaking slowly. Try to increase your verbal energy.")
        elif wpm > 170:
            st.info("🚀 **Pace:** Too fast! Practice pausing between important points.")
        
        st.markdown('</div>', unsafe_allow_html=True)