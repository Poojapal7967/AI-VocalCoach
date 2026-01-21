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

# --- 1. USER-FRIENDLY UI STYLING (Speeko Professional) ---
st.set_page_config(page_title="Speeko AI Coach", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #F7F9FC; color: #333333; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E6ED; }
    .metric-card { background: white; border-radius: 20px; padding: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #E0E6ED; margin-bottom: 25px; }
    .card-title { font-size: 1.4rem; font-weight: 700; color: #111827; margin-bottom: 15px; }
    .stButton>button { background-color: #00A3FF; color: white; border-radius: 50px; padding: 15px 30px; border: none; font-weight: 800; font-size: 18px; width: 100%; box-shadow: 0 4px 10px rgba(0, 163, 255, 0.3); }
    .feedback-row { display: flex; align-items: center; padding: 15px 0; border-bottom: 1px solid #F0F2F6; }
    .icon-box { width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. ADVANCED BACKEND LOGIC (Full Metrics) ---
def get_advanced_stats(file_path):
    y, sr = librosa.load(file_path)
    # Talk Time Balance
    intervals = librosa.effects.split(y, top_db=25)
    speech_time = sum([start_end[1] - start_end[0] for start_end in intervals]) / sr
    total_time = len(y) / sr
    talk_ratio = (speech_time / total_time) * 100 if total_time > 0 else 0

    # Intonation Logic
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
    variation = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
    intonation = "Dynamic & Energetic" if variation >= 15 else "A bit Monotone"
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)): tempo = float(tempo[0])
    return tempo, avg_pitch, intonation, talk_ratio, y, sr

def get_smart_feedback(mode, transcript, tempo):
    # FIXED: Expanded filler list for better detection
    fillers = ["um", "ummm", "ah", "ahh", "uh", "uhh", "basically", "actually", "like", "you know"]
    words = transcript.lower().replace(".", "").replace(",", "").split()
    found = [w for w in words if w in fillers]
    tips = []
    if mode == "💼 Interview": tips.append("🤝 **Pro Tip:** End your sentences firmly to show authority.")
    else: tips.append("🌟 **Impact:** Use volume variety to capture audience focus.")
    return tips, found

# --- 3. SESSION INITIALIZATION ---
if 'vocal_history' not in st.session_state: st.session_state.vocal_history = [] 
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
model_w = whisper.load_model("base")

# --- 4. SIDEBAR (Full Controls) ---
with st.sidebar:
    st.markdown("<h2 style='color:#00A3FF; font-weight:900;'>Speeko AI</h2>", unsafe_allow_html=True)
    nav = st.radio("Menu", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Script Bot"])
    st.divider()
    rec_duration = st.slider("Set recording duration (s):", 3, 60, 5) #
    st.divider()
    mode_sel = st.radio("Focus Mode:", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"])

# --- 5. MAIN MODULE: DASHBOARD ---
if nav == "🏠 Dashboard":
    st.title(f"Ready to master your {mode_sel.split()[-1]} skills?")
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown('<div class="metric-card"><p class="card-title">1. Record Your Voice</p>', unsafe_allow_html=True)
        if st.button(f"🎤 START RECORDING ({rec_duration}s)"):
            with st.status(f"Listening for {rec_duration}s..."): #
                fs = 44100
                rec = sd.rec(int(rec_duration * fs), samplerate=fs, channels=1, dtype='float32')
                sd.wait()
                write("speech.wav", fs, (rec * 32767).astype(np.int16))
                st.rerun()
        
        if os.path.exists("speech.wav"):
            st.audio("speech.wav")
            if st.button("🛑 ANALYZE MY VOICE"):
                with st.spinner("AI is listening for fillers..."):
                    # FIXED: Added fp16=False and explicit language for filler accuracy
                    res = model_w.transcribe("speech.wav", fp16=False, language='en')
                    tempo, pitch, inton, talk, y, sr = get_advanced_stats("speech.wav")
                    tips, found = get_smart_feedback(mode_sel, res['text'], tempo)
                    
                    word_count = len(res['text'].split())
                    f_perc = int((len(found) / word_count) * 100) if word_count > 0 else 0
                    
                    st.session_state.vocal_history.append({'score': 85, 'fillers': f_perc, 'talk': talk})
                    st.session_state.update({
                        'transcription': res['text'], 'tips': tips, 'filler_count': f_perc, 
                        'tempo': tempo, 'avg_pitch': pitch, 'intonation': inton, 
                        'talk_ratio': talk, 'analysis_ready': True
                    })
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="metric-card"><p class="card-title">2. Your AI Feedback</p>', unsafe_allow_html=True)
        if st.session_state.get('analysis_ready'):
            perf_data = [
                ("Talking Speed", f"{int(st.session_state.tempo)} BPM", "⏳", "#FFF4ED"),
                ("Word Clarity", f"{st.session_state.filler_count}% fillers", "💬", "#EBF5FF"), #
                ("Voice Energy", st.session_state.intonation, "🎵", "#E9FBF0"), #
                ("Speech Balance", f"{int(st.session_state.talk_ratio)}% Talk Time", "⏱️", "#FFF0F0")
            ]
            for l, v, i, b in perf_data:
                st.markdown(f'<div class="feedback-row"><div class="icon-box" style="background-color: {b};">{i}</div><div style="flex-grow: 1;"><div style="font-size: 11px; color: #8898AA;">{l}</div><div style="font-weight: 700; font-size: 16px;">{v}</div></div></div>', unsafe_allow_html=True)
        else: st.info("Record and analyze to see insights.")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get('analysis_ready'):
        st.markdown('<div class="metric-card"><p class="card-title">Confidence Visualization</p>', unsafe_allow_html=True)
        yp, srp = librosa.load("speech.wav")
        fig, ax = plt.subplots(figsize=(12, 2))
        h_color = '#00A3FF' if "Dynamic" in st.session_state.intonation else '#FF4B4B'
        librosa.display.waveshow(yp, sr=srp, ax=ax, color=h_color, alpha=0.8)
        ax.axis('off'); st.pyplot(fig)
        st.markdown(f'**Transcript:** "{st.session_state.transcription}"')
        for tip in st.session_state.tips: st.write(f"💡 {tip}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- 6. OTHER MODULES (Intact) ---
elif nav == "📈 My Progress":
    st.title("Track your speaker progress.")
    if st.session_state.vocal_history:
        st.line_chart([s['talk'] for s in st.session_state.vocal_history])
    else: st.info("Perform sessions to see trends.")

elif nav == "🤖 AI Script Bot":
    st.title("Personalized practice scripts.")
    t_in = st.text_input("What is the topic?")
    if st.button("Generate My Script ✨"):
        if t_in:
            response = ai_model.generate_content(f"Write a script for {mode_sel}: {t_in}")
            st.session_state.chat_history.append({"a": response.text}); st.rerun()
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f'<div class="metric-card" style="border-left: 5px solid #00A3FF;">{chat["a"]}</div>', unsafe_allow_html=True)