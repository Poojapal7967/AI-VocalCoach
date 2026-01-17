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
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # Replace with valid key
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-pro')
except:
    st.error("Invalid API Key. Update Gemini Key in code.")

# --- 1. SPEEKO-STYLE UI ---
st.set_page_config(page_title="Speeko AI Coach", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #F7F9FC; color: #333333; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E6ED; }
    .metric-card { background: white; border-radius: 20px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #E0E6ED; margin-bottom: 20px; }
    .metric-label { font-size: 13px; color: #8898AA; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .stButton>button { background-color: #00A3FF; color: white; border-radius: 50px; padding: 12px 25px; border: none; font-weight: bold; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background-color: #0082CC; transform: translateY(-2px); }
    .feedback-row { display: flex; align-items: center; padding: 12px 0; border-bottom: 1px solid #F0F2F6; }
    .icon-box { width: 35px; height: 35px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. BACKEND LOGIC (Smart Pitch Integration) ---
def get_vocal_stats(file_path):
    y_data, sr_data = librosa.load(file_path)
    pitches, _ = librosa.piptrack(y=y_data, sr=sr_data)
    pitch_values = pitches[pitches > 0]
    
    avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
    # Pitch Variation (Std Dev) to check modulation
    variation = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
    
    # Logic: Agar variation 15 Hz se kam hai toh Monotone
    intonation_status = "Monotone" if variation < 15 else "Dynamic"
    
    tempo, _ = librosa.beat.beat_track(y=y_data, sr=sr_data)
    if isinstance(tempo, (list, np.ndarray)): tempo = float(tempo[0])
    return tempo, avg_pitch, intonation_status, y_data, sr_data

def get_smart_feedback(mode, transcript, tempo):
    fillers = ["um", "ah", "basically", "actually", "like"]
    words = transcript.lower().split()
    found = [w for w in words if w in fillers]
    tips = []
    if mode == "💼 Interview": tips.append("🤝 **Pro Tip:** Authority comes from calm, steady finishes.")
    else: tips.append("🌟 **Impact:** Use volume variety to capture focus.")
    return tips, found

# --- 3. SESSION INITIALIZATION ---
if 'vocal_history' not in st.session_state: st.session_state.vocal_history = [] 
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

@st.cache_resource
def load_whisper(): return whisper.load_model("base")
model_w = load_whisper()

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='color:#00A3FF; font-weight:800;'>Speeko AI</h2>", unsafe_allow_html=True)
    nav = st.radio("Menu", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Script Bot"])
    st.divider()
    st.markdown("**Focus Mode:**")
    mode_sel = st.radio("", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"])

# --- 5. MAIN DASHBOARD ---
if nav == "🏠 Dashboard":
    st.title("Feedback just for you.") 
    col_l, col_r = st.columns([1.2, 1])
    
    with col_l:
        st.markdown('<div class="metric-card"><p class="metric-label">Voice Recorder</p>', unsafe_allow_html=True)
        if os.path.exists("speech.wav"): st.audio("speech.wav")
        
        c_btn1, c_btn2 = st.columns(2)
        if c_btn1.button("🎤 START RECORDING"):
            with st.status("Recording..."):
                rec = sd.rec(int(5 * 44100), samplerate=44100, channels=1, dtype='float32')
                sd.wait()
                write("speech.wav", 44100, (rec * 32767).astype(np.int16))
                st.rerun()
        
        if c_btn2.button("🛑 ANALYZE"):
            if os.path.exists("speech.wav"):
                with st.spinner("Analyzing..."):
                    res = model_w.transcribe("speech.wav")
                    tempo, pitch, intonation, y, sr = get_vocal_stats("speech.wav")
                    tips, found = get_smart_feedback(mode_sel, res['text'], tempo)
                    f_perc = int((len(found) / len(res['text'].split())) * 100) if len(res['text'].split()) > 0 else 0
                    
                    st.session_state.vocal_history.append({'score': 85, 'fillers': f_perc})
                    st.session_state.update({
                        'transcription': res['text'], 'tips': tips, 'filler_count': f_perc, 
                        'tempo': tempo, 'avg_pitch': pitch, 'intonation': intonation, 'analysis_ready': True
                    })
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="metric-card"><p class="metric-label">Vocal Performance</p>', unsafe_allow_html=True)
        metrics = [
            ("Pace", f"{int(st.session_state.get('tempo', 0))} BPM", "⏳", "#FFF4ED"),
            ("Eloquence", f"{st.session_state.get('filler_count', 0)}% fillers", "💬", "#EBF5FF"),
            ("Intonation", st.session_state.get('intonation', 'Pending'), "🎵", "#E9FBF0"),
            ("Pitch (Avg)", f"{int(st.session_state.get('avg_pitch', 0))} Hz", "🔊", "#FFF0F0")
        ]
        for label, val, icon, bg in metrics:
            st.markdown(f'<div class="feedback-row"><div class="icon-box" style="background-color: {bg};">{icon}</div><div style="flex-grow: 1;"><div style="font-size: 11px; color: #8898AA;">{label}</div><div style="font-weight: 700; font-size: 16px;">{val}</div></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get('analysis_ready'):
        st.markdown(f'<div class="metric-card"><p class="metric-label">AI Insight Report</p><p>"{st.session_state.transcription}"</p></div>', unsafe_allow_html=True)
        y_plot, sr_plot = librosa.load("speech.wav")
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.set_facecolor('#FFFFFF')
        librosa.display.waveshow(y_plot, sr=sr_plot, ax=ax, color='#00A3FF')
        ax.axis('off'); st.pyplot(fig)

elif nav == "📈 My Progress":
    st.title("Track your speaker progress.")
    if len(st.session_state.vocal_history) > 1:
        prev, now = st.session_state.vocal_history[-2], st.session_state.vocal_history[-1]
        st.metric("Clarity Improvement", f"{now['fillers']}%", f"{prev['fillers'] - now['fillers']}% Improved")
    else: st.info("Perform 2 sessions to see progress.")

elif nav == "🤖 AI Script Bot":
    st.title("Personalized practice.")
    p_in = st.text_input("What to practice?")
    if st.button("Generate Script ✨"):
        if p_in:
            response = ai_model.generate_content(f"Script for {mode_sel}: {p_in}")
            st.session_state.chat_history.append({"a": response.text}); st.rerun()
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f'<div class="metric-card" style="border-left: 5px solid #00A3FF;">{chat["a"]}</div>', unsafe_allow_html=True)