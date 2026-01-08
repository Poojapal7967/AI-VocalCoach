import streamlit as st

# --- 1. PREMIUM NEON BOXED UI (Merging image_3114e4 & image_31155e) ---
st.set_page_config(page_title="AI Vocal Coach Pro", layout="wide")

st.markdown("""
    <style>
    /* Dark Deep Theme */
    .stApp { background: #0c0c1e; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* Top Recording Bar with Glow (image_31155e) */
    .recording-bar {
        background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 10px 20px; margin-bottom: 30px;
        display: flex; align-items: center; justify-content: space-between;
        backdrop-filter: blur(10px);
    }
    .neon-line { flex-grow: 1; height: 3px; background: #ff4b4b; margin: 0 20px; position: relative; }
    .neon-dot { 
        width: 14px; height: 14px; background: #ff4b4b; border-radius: 50%; 
        position: absolute; right: 0; top: -5px; box-shadow: 0 0 15px #ff4b4b; 
    }

    /* Professional Boxed Cards (image_3114e4 structure + Neon effects) */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px; padding: 40px; text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px); height: 260px;
        display: flex; flex-direction: column; justify-content: center;
        transition: 0.3s;
    }
    .metric-card:hover { transform: translateY(-5px); background: rgba(255, 255, 255, 0.08); }

    /* Neon Borders from image_31155e */
    .border-red { border-top: 4px solid #ff4b4b; box-shadow: 0 -5px 15px rgba(255, 75, 75, 0.2); }
    .border-orange { border-top: 4px solid #ff9800; box-shadow: 0 -5px 15px rgba(255, 152, 0, 0.2); }
    .border-blue { border-top: 4px solid #00f2fe; box-shadow: 0 -5px 15px rgba(0, 242, 254, 0.2); }

    /* Typography */
    .big-num { font-size: 65px; font-weight: 800; color: #ffffff; line-height: 1; margin-bottom: 10px; }
    .label-sub { font-size: 14px; color: #8e949a; text-transform: uppercase; letter-spacing: 2px; }

    /* Start Recording Neon Button (image_31155e Exact Style) */
    div.stButton > button {
        background: transparent !important; color: #ffffff !important;
        border: 2px solid #ff4b4b !important; border-radius: 30px !important;
        padding: 12px 35px !important; font-weight: 700 !important;
        box-shadow: 0px 0px 20px rgba(255, 75, 75, 0.5); 
        width: 250px !important; margin-top: 25px; transition: 0.3s;
    }
    div.stButton > button:hover { background: #ff4b4b !important; box-shadow: 0px 0px 30px #ff4b4b; }

    /* Central WAV Box */
    .wav-container { background: #1a1c1e; border-radius: 10px; padding: 15px; border: 1px solid #333; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. TOP AUDIO BAR SECTION ---
st.caption("Recording Time (seconds)")
st.markdown("""
    <div class="recording-bar">
        <span>▶ 0:00 / 0:05</span>
        <div class="neon-line"><div class="neon-dot"></div></div>
        <span>🔊 ⋮</span>
    </div>
""", unsafe_allow_html=True)

# --- 3. PERFORMANCE GRID (image_3114e4 structure + Premium Effects) ---
st.markdown('<p style="font-size:18px; font-weight:600; margin-bottom:25px; color:#ffcc00;">📊 PERFORMANCE REPORT</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
        <div class="metric-card border-red">
            <div class="big-num">19</div>
            <div class="label-sub">Confidence Score</div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
        <div class="metric-card border-orange">
            <div class="wav-container">🔊 speech.wav</div>
            <div class="label-sub" style="margin-top:20px;">Transbich.wav Ready</div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
        <div class="metric-card border-blue">
            <div class="big-num">3%</div>
            <div class="label-sub">Fillers Found</div>
            <div style="font-size:13px; color:#8e949a; margin-top:15px; font-style:italic;">Speaking: Normal</div>
        </div>
    """, unsafe_allow_html=True)

# Start Recording Button at the bottom left
if st.button("🔴 START RECORDING"):
    st.toast("Capturing high-fidelity audio...")

# --- 4. COACHING TIPS (Premium Look) ---
st.write("##")
st.markdown("""
    <div style="background: rgba(0, 242, 254, 0.05); border-left: 5px solid #00f2fe; border-radius: 12px; padding: 25px;">
        <p style="color:#00f2fe; font-weight:bold; font-size:18px; margin-bottom:15px;">💡 COACHING FEEDBACK</p>
        <div style="margin-bottom:12px;">⭐ <span style="color:#4caf50;">Success: Your vocal tone is very stable today.</span></div>
        <div style="margin-bottom:12px;">⭐ <span style="color:#ff9800;">Warning: Detected slight filler word frequency at 0:03.</span></div>
        <div style="margin-bottom:12px;">⭐ <span style="color:#f44336;">Speed: Pace was high during the opening sentence.</span></div>
    </div>
""", unsafe_allow_html=True)