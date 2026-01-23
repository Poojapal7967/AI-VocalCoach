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

# --- 1. PAGE STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- 2. PREMIUM UI STYLING (SaaS Professional) ---
st.set_page_config(page_title="Speeko AI Coach", layout="wide")
st.markdown("""
    <style>
    /* Premium Color Palette & Background */
    .stApp { background-color: #F8FAFC; color: #1E293B; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #F1F5F9; }
    
    /* Classy Hero Section */
    .hero-container {
        padding: 100px 20px;
        text-align: center;
        background: radial-gradient(circle at top right, #EFF6FF, #F8FAFC);
        border-radius: 40px;
        margin-bottom: 60px;
        border: 1px solid rgba(0, 163, 255, 0.05);
    }
    .classy-title {
        font-size: 4.5rem;
        font-weight: 900;
        color: #0F172A;
        letter-spacing: -2px;
        line-height: 1.1;
        margin-bottom: 20px;
    }
    .highlight-blue {
        background: linear-gradient(90deg, #00A3FF, #0066FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Glassmorphism Feature Cards */
    .premium-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 24px;
        padding: 40px 32px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
        height: 100%;
    }
    .premium-card:hover {
        transform: translateY(-12px);
        box-shadow: 0 20px 40px rgba(0, 163, 255, 0.08);
        border-color: #00A3FF;
    }
    
    /* Dashboard Elements */
    .metric-card { 
        background: white; border-radius: 20px; padding: 30px; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.03); border: 1px solid #F1F5F9; 
        margin-bottom: 25px; 
    }
    .stButton>button { 
        background-color: #00A3FF; color: white; border-radius: 12px; 
        padding: 14px 28px; border: none; font-weight: 700; font-size: 16px; 
        width: 100%; transition: 0.3s ease;
    }
    .stButton>button:hover { background-color: #0082CC; transform: translateY(-2px); }
    </style>
""", unsafe_allow_html=True)

# --- 3. BACKEND LOGIC ---
def get_advanced_stats(file_path):
    y, sr = librosa.load(file_path)
    intervals = librosa.effects.split(y, top_db=25)
    speech_time = sum([start_end[1] - start_end[0] for start_end in intervals]) / sr
    talk_ratio = (speech_time / (len(y)/sr)) * 100 if len(y) > 0 else 0
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    variation = float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    intonation = "Dynamic & Energetic" if variation >= 15 else "A bit Monotone"
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)): tempo = float(tempo[0])
    return tempo, np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0, intonation, talk_ratio, y, sr

def get_smart_feedback(mode, transcript, tempo):
    fillers = ["um", "ummm", "ah", "ahh", "uh", "uhh", "basically", "actually", "like", "you know"]
    words = transcript.lower().replace(".", "").replace(",", "").split()
    found = [w for w in words if w in fillers]
    tips = ["🌟 **Impact:** Use volume variety to capture audience focus."]
    if mode == "💼 Interview": tips = ["🤝 **Pro Tip:** End your sentences firmly to show authority."]
    return tips, found

# --- 4. NAVIGATION LOGIC ---
if st.session_state.page == 'home':
    # --- CLASSY LANDING PAGE ---
    st.markdown("""
        <div class="hero-container">
            <h1 class="classy-title">Speak <span class="highlight-blue">Confidently.</span></h1>
            <p style="font-size: 1.3rem; color: #64748B; max-width: 750px; margin: 0 auto 40px auto; line-height: 1.6;">
                The world's most intuitive AI speech coach. Master your tone, eliminate fillers, and find your perfect voice with precision analytics.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    _, col_btn, _ = st.columns([1, 0.7, 1])
    with col_btn:
        if st.button("🚀 Start Free Coaching Session"):
            st.session_state.page = 'dashboard'
            st.rerun()
            
    st.write("##")
    st.divider()
    
    # Premium Feature Cards
    f1, f2, f3 = st.columns(3)
    card_data = [
        ("📊", "Real-time Insights", "Get precise feedback on your pace, fillers, and vocal energy instantly."),
        ("🤖", "AI Practice Bot", "Generate high-impact practice scripts tailored for interviews or keynotes."),
        ("📈", "Progress Tracking", "Visualize your growth over time with our proprietary confidence scoring.")
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3], card_data):
        col.markdown(f"""
            <div class="premium-card">
                <div style="font-size: 48px; margin-bottom: 20px;">{icon}</div>
                <h3 style="color: #0F172A; font-size: 1.5rem; font-weight: 700;">{title}</h3>
                <p style="color: #64748B; line-height: 1.6;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)

else:
    # --- DASHBOARD PAGE ---
    if 'vocal_history' not in st.session_state: st.session_state.vocal_history = [] 
    if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    model_w = whisper.load_model("base")

    with st.sidebar:
        if st.button("⬅️ Back to Home"):
            st.session_state.page = 'home'
            st.rerun()
        st.divider()
        st.markdown("<h2 style='color:#00A3FF; font-weight:900;'>Speeko AI</h2>", unsafe_allow_html=True)
        nav = st.radio("Menu", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Script Bot"])
        st.divider()
        rec_duration = st.slider("Recording duration (s):", 3, 60, 5) #
        mode_sel = st.radio("Focus Mode:", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"])

    if nav == "🏠 Dashboard":
        st.title(f"Master your {mode_sel.split()[-1]} skills.") # Classy Title
        col_l, col_r = st.columns([1, 1])
        
        with col_l:
            st.markdown('<div class="metric-card"><p class="card-title">1. Voice Capture</p>', unsafe_allow_html=True)
            if st.button(f"🎤 START RECORDING ({rec_duration}s)"):
                with st.status("Analyzing voice..."):
                    fs = 44100
                    rec = sd.rec(int(rec_duration * fs), samplerate=fs, channels=1, dtype='float32')
                    sd.wait()
                    write("speech.wav", fs, (rec * 32767).astype(np.int16))
                    st.rerun()
            if os.path.exists("speech.wav"):
                st.audio("speech.wav")
                if st.button("🛑 ANALYZE CLIP"):
                    with st.spinner("Processing speech metrics..."):
                        res = model_w.transcribe("speech.wav", fp16=False, language='en')
                        tempo, pitch, inton, talk, y, sr = get_advanced_stats("speech.wav")
                        tips, found = get_smart_feedback(mode_sel, res['text'], tempo)
                        word_count = len(res['text'].split())
                        f_perc = int((len(found) / word_count) * 100) if word_count > 0 else 0
                        st.session_state.vocal_history.append({'score': 85, 'fillers': f_perc, 'talk': talk})
                        st.session_state.update({'transcription': res['text'], 'tips': tips, 'filler_count': f_perc, 'tempo': tempo, 'intonation': inton, 'talk_ratio': talk, 'analysis_ready': True})
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="metric-card"><p class="card-title">2. Analysis Breakdown</p>', unsafe_allow_html=True)
            if st.session_state.get('analysis_ready'):
                perf_data = [
                    ("Talking Speed", f"{int(st.session_state.tempo)} BPM", "⏳", "#FFF7ED"),
                    ("Word Clarity", f"{st.session_state.filler_count}% fillers", "💬", "#EFF6FF"),
                    ("Voice Energy", st.session_state.intonation, "🎵", "#F0FDF4"),
                    ("Speech Balance", f"{int(st.session_state.talk_ratio)}% Talk", "⏱️", "#FEF2F2")
                ]
                for l, v, i, b in perf_data:
                    st.markdown(f'<div class="feedback-row" style="display:flex; align-items:center; padding:12px 0; border-bottom:1px solid #F1F5F9;"><div style="width:40px; height:40px; background:{b}; border-radius:10px; display:flex; align-items:center; justify-content:center; margin-right:15px; font-size:18px;">{i}</div><div><div style="font-size:11px; color:#64748B;">{l}</div><div style="font-weight:700; font-size:16px;">{v}</div></div></div>', unsafe_allow_html=True)
            else: st.info("Analysis will appear here.")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('analysis_ready'):
            st.markdown('<div class="metric-card"><p class="card-title">Confidence Visualization</p>', unsafe_allow_html=True)
            yp, srp = librosa.load("speech.wav")
            fig, ax = plt.subplots(figsize=(12, 2))
            h_color = '#00A3FF' if "Dynamic" in st.session_state.intonation else '#EF4444'
            librosa.display.waveshow(yp, sr=srp, ax=ax, color=h_color, alpha=0.8)
            ax.axis('off'); st.pyplot(fig) #
            st.markdown(f'**Transcript:** "{st.session_state.transcription}"')
            for tip in st.session_state.tips: st.write(f"💡 {tip}")
            st.markdown('</div>', unsafe_allow_html=True)