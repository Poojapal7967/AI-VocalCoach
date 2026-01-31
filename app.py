import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import pandas as pd 
import google.generativeai as genai 

# --- 0. AI CONFIG ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-pro')
except:
    st.error("Invalid API Key. Please update in the code.")

# --- 1. STATE MANAGEMENT ---
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'vocal_history' not in st.session_state: st.session_state.vocal_history = [] 

# --- 2. OPTIMIZED MODEL LOADING ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# --- 3. ELITE NEON DARK UI STYLING ---
st.set_page_config(page_title="Speeko Elite", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #05070A; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    
    /* 1. Shimmering Title */
    .shimmer-title {
        font-size: 5.5rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #FF00CC, #3333FF, #00FFCC, #FF00CC);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        filter: drop-shadow(0 0 20px rgba(108, 92, 231, 0.4));
    }
    @keyframes shine { to { background-position: 200% center; } }

    /* 2. Visual Card Styling */
    .visual-card {
        background: rgba(13, 17, 23, 0.6);
        backdrop-filter: blur(12px);
        padding: 40px; border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%; text-align: center;
    }
    .visual-card:hover {
        transform: translateY(-15px);
        border-color: #00FFCC;
        box-shadow: 0 20px 50px rgba(0, 255, 204, 0.2);
    }

    /* 3. Laser-Shine Buttons */
    .stButton>button { 
        background: linear-gradient(90deg, #FF00CC, #3333FF) !important; color: white !important; 
        border-radius: 50px !important; padding: 15px 40px !important; border: none !important; 
        font-weight: 800; transition: 0.4s; width: 100%; height: 60px;
        position: relative; overflow: hidden; font-size: 1.1rem;
    }
    .stButton>button::before {
        content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: 0.6s;
    }
    .stButton>button:hover::before { left: 100%; }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0 10px 30px rgba(255, 0, 204, 0.4); }
    
    .metric-card { background: #0D1117; border-radius: 20px; padding: 25px; border: 1px solid #30363D; margin-bottom: 20px; }
    .feedback-box { background: rgba(108, 92, 231, 0.1); border-left: 5px solid #6C5CE7; padding: 15px; border-radius: 10px; margin-top: 10px; }
    
    [data-testid="stSidebarNav"] span, .stRadio label { display: flex !important; align-items: center !important; gap: 10px !important; }
    .insight-row { display: flex; align-items: center; margin-bottom: 15px; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

# --- 4. CORE ANALYTICS ENGINE (Functions Intact) ---
def get_vocal_analysis(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    variation = float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
    energy_label = "🔥 High Impact" if variation >= 18 else "⚡ Moderate"
    intervals = librosa.effects.split(y, top_db=25)
    speech_time = sum([s[1] - s[0] for s in intervals]) / sr
    talk_ratio = (speech_time / (len(y)/sr)) * 100 if len(y) > 0 else 0
    return int(tempo[0] if isinstance(tempo, np.ndarray) else tempo), energy_label, int(talk_ratio), y, sr

def get_coaching_feedback(pace, fillers, balance, mode):
    feedback_list = []
    if pace > 165: feedback_list.append("🚀 **Slow Down:** Your pace is too fast.")
    elif pace < 100: feedback_list.append("⏳ **Speed Up:** You're a bit slow.")
    if fillers > 2: feedback_list.append(f"💬 **Fillers:** Detected {fillers} words.")
    mode_tips = {"🎤 Public Speaking": "🌟 Stage Tip: Project your voice.", "🎧 Anchoring": "🎙️ Studio Tip: Resonate tones.", "💼 Interview": "🤝 Rapport Tip: Mirror energy.", "👨‍🏫 Teaching": "🍎 Edu Tip: Pause after key points."}
    feedback_list.append(mode_tips.get(mode, ""))
    return feedback_list

# --- 5. THE NEW VISUAL LANDING PAGE ---
if st.session_state.page == 'home':
    # --- HERO SECTION ---
    st.markdown('<div style="text-align:center; padding:80px 0;">', unsafe_allow_html=True)
    st.markdown('<h1 class="shimmer-title">Speeko Elite</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1.8rem; opacity:0.8; letter-spacing:2px; font-weight:300;">THE FUTURE OF VOCAL INTELLIGENCE</p>', unsafe_allow_html=True)
    
    _, c_btn, _ = st.columns([1,1,1])
    with c_btn:
        if st.button("🚀 LAUNCH DASHBOARD"):
            st.session_state.page = 'dashboard'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- IMAGE & FEATURE SHOWCASE ---
    st.markdown("---")
    col_img, col_txt = st.columns([1.2, 1])
    with col_img:
        # High-end AI Voice Analytics Visual
        st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop", 
                 caption="Speeko Neural Interface v2.0")
    with col_txt:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🧠 Why Speeko Elite?")
        st.markdown("""
        * **Neural Pitch Mapping:** Proprietary algorithms track your vocal frequency in real-time.
        * **AI-Generated Scripts:** Never run out of things to say with Gemini-powered synthesis.
        * **Biometric Insights:** Track energy modulation and 'filler-to-flow' ratios.
        * **Persistent Evolution:** Secure history vault to watch your growth from amateur to master.
        """)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- THREE PILLAR SECTION ---
    st.markdown("## 💎 System Architecture")
    p1, p2, p3 = st.columns(3)
    pillars = [
        ("📊", "Deep Data", "Full Waveform analysis and BPM tracking for every syllable spoken."),
        ("🤖", "Logic Layer", "Whisper AI transcription coupled with Gemini for contextual feedback."),
        ("📉", "Analytics", "Interactive charts showing your progress over days, weeks, and months.")
    ]
    for col, (icon, title, desc) in zip([p1, p2, p3], pillars):
        col.markdown(f"""
            <div class="visual-card">
                <h1 style="font-size:3rem;">{icon}</h1>
                <h3>{title}</h3>
                <p style="opacity:0.7;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; opacity: 0.3; padding: 30px; border-top: 1px solid rgba(255,255,255,0.1);">
            <p>© 2026 Speeko Elite AI | Precision Communication Suite</p>
        </div>
    """, unsafe_allow_html=True)

# --- 6. DASHBOARD LOGIC (Intact & Functional) ---
else:
    with st.sidebar:
        st.markdown("""<div style="background:linear-gradient(90deg, #FF00CC, #3333FF); padding:10px; border-radius:10px; text-align:center;"><h3 style="margin:0;">Speeko Elite</h3></div>""", unsafe_allow_html=True)
        if st.button("⬅️ BACK TO HOME"):
            st.session_state.page = 'home'
            st.rerun()
        st.divider()
        nav = st.radio("System Menu", ["🏠 Dashboard", "📈 Growth History", "🤖 AI Script Bot"])
        rec_dur = st.slider("Duration (s):", 5, 60, 10)
        mode_sel = st.selectbox("Select Mode:", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"])

    if nav == "🏠 Dashboard":
        st.title(f"Coaching Engine: {mode_sel}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card"><h4>1. ANALYZE VOICE</h4>', unsafe_allow_html=True)
            if st.button("🎤 START CAPTURE"):
                try:
                    fs = 44100
                    rec = sd.rec(int(rec_dur * fs), samplerate=fs, channels=1, dtype='float32')
                    with st.status("Listening..."): sd.wait()
                    write("speech.wav", fs, (rec * 32767).astype(np.int16))
                    st.rerun()
                except Exception as e: st.error(f"Mic Error: {e}")

            if os.path.exists("speech.wav"):
                st.audio("speech.wav")
                if st.button("🛑 PROCESS AUDIO"):
                    with st.spinner("AI Processing..."):
                        model_w = load_whisper_model()
                        res = model_w.transcribe("speech.wav", language='en')
                        pace, energy, balance, y, sr = get_vocal_analysis("speech.wav")
                        fillers = sum([1 for w in res['text'].lower().split() if w in ["um", "ah", "uh", "like"]])
                        coach_data = get_coaching_feedback(pace, fillers, balance, mode_sel)
                        st.session_state.vocal_history.append({'Date': pd.Timestamp.now().strftime('%H:%M'), 'Pace': pace, 'Balance': balance, 'Mode': mode_sel})
                        st.session_state.update({'trans': res['text'], 'pace': pace, 'energy': energy, 'balance': balance, 'fillers': fillers, 'coach_data': coach_data, 'ready': True})
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card"><h4>2. AI DIAGNOSTICS</h4>', unsafe_allow_html=True)
            if st.session_state.get('ready'):
                st.write(f"⏳ **Pace:** {st.session_state.pace} BPM")
                st.write(f"🔥 **Energy:** {st.session_state.energy}")
                st.write(f"⏱️ **Balance:** {st.session_state.balance}% Vocal")
                st.write(f"💬 **Fillers:** {st.session_state.fillers}")
                st.markdown("---")
                for item in st.session_state.coach_data:
                    if item: st.markdown(f'<div class="feedback-box">{item}</div>', unsafe_allow_html=True)
            else: st.info("Initiate capture to see diagnostics.")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('ready'):
            st.markdown('<div class="metric-card"><h4>WAVEFORM ANALYSIS</h4>', unsafe_allow_html=True)
            y_plot, sr_plot = librosa.load("speech.wav")
            fig, ax = plt.subplots(figsize=(10, 2), facecolor='#0D1117')
            librosa.display.waveshow(y_plot, sr=sr_plot, ax=ax, color='#FF00CC', alpha=0.8)
            ax.axis('off'); st.pyplot(fig)
            st.write(f"**Transcript:** {st.session_state.trans}")
            st.markdown('</div>', unsafe_allow_html=True)

    elif nav == "📈 Growth History":
        st.title("Growth Analytics")
        if not st.session_state.vocal_history: st.warning("No data yet.")
        else:
            df = pd.DataFrame(st.session_state.vocal_history)
            st.line_chart(df.set_index('Date')['Pace'])
            st.table(df)

    elif nav == "🤖 AI Script Bot":
        st.title("AI Speech Writer")
        topic = st.text_input("Enter topic:")
        if st.button("GENERATE ✨"):
            res = ai_model.generate_content(f"Write a professional {mode_sel} script about: {topic}")
            st.markdown(f'<div class="metric-card">{res.text}</div>', unsafe_allow_html=True)