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

# --- 3. ELITE DARK UI STYLING ---
st.set_page_config(page_title="Speeko Elite", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #05070A; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    
    /* Hero Section */
    .hero-box {
        background: radial-gradient(circle at 70% 30%, rgba(108, 92, 231, 0.15) 0%, rgba(5, 7, 10, 1) 70%);
        padding: 80px 40px; border-radius: 30px; margin-bottom: 50px; text-align: center;
    }
    
    /* Stylish Buttons */
    .stButton>button { 
        background: linear-gradient(90deg, #FF00CC, #3333FF); color: white; border-radius: 12px; 
        padding: 12px 30px; border: none; font-weight: 800; transition: 0.3s; width: 100%;
        display: flex; align-items: center; justify-content: center; gap: 8px; height: 50px;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 10px 20px rgba(108, 92, 231, 0.3); }
    
    /* Cards & Layouts */
    .feature-card {
        background: rgba(255, 255, 255, 0.03); padding: 30px; border-radius: 20px; 
        border: 1px solid rgba(255, 255, 255, 0.1); height: 100%; transition: 0.3s;
    }
    .feature-card:hover { background: rgba(255, 255, 255, 0.05); border-color: #6C5CE7; }
    
    .metric-card { background: #0D1117; border-radius: 20px; padding: 25px; border: 1px solid #30363D; margin-bottom: 20px; }
    .feedback-box { background: rgba(108, 92, 231, 0.1); border-left: 5px solid #6C5CE7; padding: 15px; border-radius: 10px; margin-top: 10px; }
    
    /* Sidebar Fixes */
    [data-testid="stSidebarNav"] span, .stRadio label { display: flex !important; align-items: center !important; gap: 10px !important; }
    .insight-row { display: flex; align-items: center; margin-bottom: 15px; font-size: 1.1rem; }
    .insight-icon { margin-right: 12px; width: 25px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- 4. CORE ANALYTICS ENGINE ---
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
    if pace > 165: feedback_list.append("🚀 **Slow Down:** Your pace is too fast. Use pauses to improve clarity.")
    elif pace < 100: feedback_list.append("⏳ **Speed Up:** You're a bit slow. Increase energy to keep the audience engaged.")
    if fillers > 2: feedback_list.append(f"💬 **Reduce Fillers:** Detected {fillers} filler words. Try using silent pauses.")
    if balance < 75: feedback_list.append("⏱️ **Flow:** Too many silent gaps. Try to maintain a consistent stream of thought.")
    
    mode_tips = {
        "🎤 Public Speaking": "🌟 Stage Tip: Project your voice further for impact.",
        "🎧 Anchoring": "🎙️ Studio Tip: Focus on resonant tones and crisp articulation.",
        "💼 Interview": "🤝 Rapport Tip: Mirror the interviewer's energy level.",
        "👨‍🏫 Teaching": "🍎 Edu Tip: Pause for 3 seconds after key points for retention."
    }
    feedback_list.append(mode_tips.get(mode, ""))
    return feedback_list

# --- 5. NAVIGATION ENGINE ---
if st.session_state.page == 'home':
    # --- HERO SECTION ---
    st.markdown("""
        <div class="hero-box">
            <h1 style="font-size:4.5rem; font-weight:900; margin-bottom:10px;">
                Speak <span style="color: #6C5CE7; text-shadow: 0 0 20px rgba(108,92,231,0.5);"><</span>Confidently.
            </h1>
            <p style="font-size:1.4rem; opacity:0.8;">The AI Vocal Coach for the Next Generation of Communicators.</p>
        </div>
    """, unsafe_allow_html=True)
    
    c_btn1, c_btn2, c_btn3 = st.columns([1,1,1])
    with c_btn2:
        if st.button("🚀 LAUNCH DASHBOARD"):
            st.session_state.page = 'dashboard'
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    # --- KEY BENEFITS SECTION ---
    st.markdown("## 💎 Premium Features")
    b1, b2, b3 = st.columns(3)
    benefits = [
        ("📊", "Deep Analytics", "Track BPM, energy modulation, and talk-to-silence ratios with precision."),
        ("🧠", "AI Coaching", "Get mode-specific feedback for interviews, teaching, or public speaking."),
        ("📉", "Persistent Growth", "Visual history charts help you monitor your vocal evolution over time.")
    ]
    for col, (icon, title, desc) in zip([b1, b2, b3], benefits):
        col.markdown(f"""
            <div class="feature-card">
                <h1>{icon}</h1>
                <h3>{title}</h3>
                <p style="opacity:0.7;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- HOW IT WORKS (Scroll Content) ---
    st.markdown("## 🛠️ Your Path to Mastery")
    s1, s2, s3 = st.columns(3)
    steps = [
        ("01", "Record", "Choose your practice mode and record your speech in high fidelity."),
        ("02", "Analyze", "Our Whisper-powered AI dissects your pace, fillers, and clarity."),
        ("03", "Improve", "Apply targeted feedback and repeat to conquer stage fright.")
    ]
    for col, (num, title, desc) in zip([s1, s2, s3], steps):
        col.markdown(f"""
            <div style="border-left: 2px solid #6C5CE7; padding-left: 20px;">
                <h2 style="color: #6C5CE7; margin-bottom:0;">{num}</h2>
                <h4>{title}</h4>
                <p style="opacity:0.6;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    # --- SPECIALIZED MODES SHOWCASE ---
    st.markdown("## 🎭 Tailored for Every Scenario")
    m1, m2, m3, m4 = st.columns(4)
    modes = [
        ("🎤", "Public Speaking", "Master the stage"),
        ("🎧", "Anchoring", "Crisp studio voice"),
        ("💼", "Interview", "Rapport & Confidence"),
        ("👨‍🏫", "Teaching", "Clarity & Pacing")
    ]
    for col, (icon, title, subtitle) in zip([m1, m2, m3, m4], modes):
        col.markdown(f"""
            <div style="text-align: center; background: rgba(108,92,231,0.05); padding: 20px; border-radius: 15px;">
                <h1>{icon}</h1>
                <h5 style="margin-bottom:0;">{title}</h5>
                <small style="opacity:0.5;">{subtitle}</small>
            </div>
        """, unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; opacity: 0.3; padding: 20px; border-top: 1px solid rgba(255,255,255,0.1);">
            <p>© 2026 Speeko Elite AI | Precision Communication Tool</p>
        </div>
    """, unsafe_allow_html=True)

else:
    # --- DASHBOARD & OTHER PAGES (Same logic as current code) ---
    with st.sidebar:
        st.markdown("""<div style="background:linear-gradient(90deg, #FF00CC, #3333FF); padding:10px; border-radius:10px; text-align:center;"><h3 style="margin:0;">Speeko Elite</h3></div>""", unsafe_allow_html=True)
        if st.button("⬅️ HOME"):
            st.session_state.page = 'home'
            st.rerun()
        st.divider()
        nav = st.radio("Menu", ["🏠 Dashboard", "📈 My Progress", "🤖 AI Script Bot"])
        rec_dur = st.slider("Duration (s):", 5, 60, 10)
        mode_sel = st.selectbox("Focus Mode:", ["🎤 Public Speaking", "🎧 Anchoring", "💼 Interview", "👨‍🏫 Teaching"])

    if nav == "🏠 Dashboard":
        st.title(f"Coaching: {mode_sel}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-card"><h4>1. RECORD PERFORMANCE</h4>', unsafe_allow_html=True)
            if st.button("🎤 START RECORDING"):
                try:
                    fs = 44100
                    rec = sd.rec(int(rec_dur * fs), samplerate=fs, channels=1, dtype='float32')
                    with st.status("Recording..."): sd.wait()
                    write("speech.wav", fs, (rec * 32767).astype(np.int16))
                    st.rerun()
                except Exception as e: st.error(f"Mic Error: {e}")

            if os.path.exists("speech.wav"):
                st.audio("speech.wav")
                b_ana, b_del = st.columns(2, gap="medium")
                with b_ana:
                    if st.button("🛑 ANALYZE & STORE"):
                        with st.spinner("AI Evaluating..."):
                            model_w = load_whisper_model()
                            res = model_w.transcribe("speech.wav", language='en')
                            pace, energy, balance, y, sr = get_vocal_analysis("speech.wav")
                            fillers = sum([1 for w in res['text'].lower().split() if w in ["um", "ah", "uh", "like"]])
                            coach_data = get_coaching_feedback(pace, fillers, balance, mode_sel)
                            st.session_state.vocal_history.append({'Date': pd.Timestamp.now().strftime('%H:%M'), 'Pace': pace, 'Balance': balance, 'Mode': mode_sel})
                            st.session_state.update({
                                'trans': res['text'], 'pace': pace, 'energy': energy, 
                                'balance': balance, 'fillers': fillers, 
                                'coach_data': coach_data, 'ready': True
                            })
                            st.rerun()
                with b_del:
                    st.markdown('<div class="delete-container">', unsafe_allow_html=True)
                    if st.button("🗑️ DELETE"):
                        if os.path.exists("speech.wav"): os.remove("speech.wav")
                        st.session_state.ready = False
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-card"><h4>2. AI INSIGHTS</h4>', unsafe_allow_html=True)
            if st.session_state.get('ready'):
                st.markdown(f'<div class="insight-row"><span class="insight-icon">⏳</span><b>Speed:</b> {st.session_state.pace} BPM</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-row"><span class="insight-icon">🔥</span><b>Energy:</b> {st.session_state.energy}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-row"><span class="insight-icon">⏱️</span><b>Balance:</b> {st.session_state.balance}% Talk</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="insight-row"><span class="insight-icon">💬</span><b>Fillers:</b> {st.session_state.fillers} detected</div>', unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("##### 🚀 Areas to Improve:")
                for item in st.session_state.coach_data:
                    if item: st.markdown(f'<div class="feedback-box">{item}</div>', unsafe_allow_html=True)
            else: st.info("Record a session to see metrics.")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('ready'):
            st.markdown('<div class="metric-card"><h4 style="margin-bottom:20px;">WAVEFORM ANALYSIS</h4>', unsafe_allow_html=True)
            y_plot, sr_plot = librosa.load("speech.wav")
            fig, ax = plt.subplots(figsize=(10, 2), facecolor='#0D1117')
            librosa.display.waveshow(y_plot, sr=sr_plot, ax=ax, color='#FF00CC', alpha=0.8)
            ax.axis('off'); st.pyplot(fig)
            st.write(f"**Transcript:** {st.session_state.trans}")
            st.markdown('</div>', unsafe_allow_html=True)

    elif nav == "📈 My Progress":
        st.title("Growth Analytics")
        if not st.session_state.vocal_history: st.warning("Analyze a recording on the Dashboard first!")
        else:
            df = pd.DataFrame(st.session_state.vocal_history)
            c_a, c_b = st.columns(2)
            with c_a: st.line_chart(df.set_index('Date')['Pace'])
            with c_b: st.bar_chart(df.set_index('Date')['Balance'])
            st.table(df)

    elif nav == "🤖 AI Script Bot":
        st.title("AI Speech Writer")
        topic = st.text_input("Enter topic:")
        if st.button("Generate Script ✨"):
            res = ai_model.generate_content(f"Write a professional {mode_sel} script about: {topic}")
            st.markdown(f'<div class="metric-card">{res.text}</div>', unsafe_allow_html=True)