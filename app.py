import streamlit as st

# --- 1. PRECISE UI STYLING (Pixel Perfect Match) ---
st.set_page_config(page_title="AI Vocal Coach Pro", layout="wide")

st.markdown("""
    <style>
    /* Background & Global Font */
    .stApp { background: #1a1c1e; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* Top Recording Time Bar (image_30f755.jpg) */
    .recording-bar {
        background: #232629; border: 1px solid #363a3f;
        border-radius: 12px; padding: 10px 20px; margin-bottom: 25px;
        display: flex; align-items: center; justify-content: space-between;
    }
    .waveform-line { flex-grow: 1; height: 3px; background: #ff4b4b; margin: 0 20px; position: relative; }
    .waveform-dot { 
        width: 14px; height: 14px; background: #ff4b4b; border-radius: 50%; 
        position: absolute; right: 0; top: -5px; box-shadow: 0 0 12px #ff4b4b; 
    }

    /* Main Performance Report Container (image_310d08.jpg) */
    .report-main {
        background: #232629; border: 1px solid #363a3f;
        border-radius: 20px; padding: 30px; margin-top: 10px;
    }

    /* Metric Grid - Three Equal Cards */
    .metrics-container {
        display: flex; justify-content: space-between; gap: 20px; margin-bottom: 30px;
    }

    .m-card {
        background: #2a2d32; border: 1px solid #3c4148;
        border-radius: 15px; padding: 25px; text-align: center; flex: 1;
        min-height: 180px; display: flex; flex-direction: column; justify-content: center;
    }

    /* Big Numeric Values */
    .value-text { font-size: 58px; font-weight: 700; color: #ffffff; line-height: 1; margin-bottom: 8px; }
    .label-text { font-size: 14px; color: #8e949a; }

    /* START RECORDING NEON BUTTON (The Red Glow) */
    div.stButton > button {
        background: transparent !important; color: #ffffff !important;
        border: 2px solid #ff4b4b !important; border-radius: 30px !important;
        padding: 12px 25px !important; font-weight: 600 !important;
        box-shadow: 0px 0px 18px rgba(255, 75, 75, 0.45); 
        width: 100% !important; margin-top: 15px;
    }

    /* Speech.wav Inner Box */
    .inner-wav-box {
        background: #1a1c1e; border: 1px solid #333;
        border-radius: 12px; padding: 15px; display: flex; align-items: center; justify-content: center;
    }

    /* COACHING TIPS (image_310d08.jpg) */
    .coaching-section { margin-top: 30px; }
    .tips-header-box { 
        background: #2a2d32; border-radius: 10px; padding: 12px 20px; 
        border-bottom: 2px solid #00f2fe; margin-bottom: 20px;
        display: flex; align-items: center;
    }
    .tip-line { margin-bottom: 12px; display: flex; align-items: center; font-size: 15px; }
    .star { font-size: 20px; margin-right: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. RECORDING TIME BAR ---
st.caption("Recording Time (seconds)")
st.markdown("""
    <div class="recording-bar">
        <span>▶ 0:00 / 0:05</span>
        <div class="waveform-line"><div class="waveform-dot"></div></div>
        <span>🔊 ⋮</span>
    </div>
""", unsafe_allow_html=True)

# --- 3. PERFORMANCE REPORT SECTION (Exact Image Match) ---
st.markdown('<div class="report-main">', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px; font-weight:600; margin-bottom:25px;">☗ Performance Report</p>', unsafe_allow_html=True)

# Three Metric Columns
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="m-card">', unsafe_allow_html=True)
    st.markdown('<div class="value-text">19</div>', unsafe_allow_html=True)
    st.markdown('<div class="label-text">Confidence Score</div>', unsafe_allow_html=True)
    if st.button("🔴 Start Recording"):
        st.toast("Capturing voice...")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="m-card" style="border-top: 2px solid #ff9800;">', unsafe_allow_html=True)
    st.markdown('<div class="inner-wav-box">🔊 speech.wav</div>', unsafe_allow_html=True)
    st.markdown('<div class="label-text" style="margin-top:15px;">transbich.wav ready</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="m-card" style="border-top: 2px solid #00f2fe;">', unsafe_allow_html=True)
    st.markdown('<div class="value-text">3%</div>', unsafe_allow_html=True)
    st.markdown('<div class="label-text">Fillers Found</div>', unsafe_allow_html=True)
    st.markdown('<p class="label-text" style="margin-top:20px;">Speaking: Normal</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- 4. COACHING TIPS SECTION (image_310d08.jpg) ---
st.markdown('<div class="coaching-section">', unsafe_allow_html=True)
st.write("What you said:")
st.markdown("""
    <div class="tips-header-box">💡 Coaching Tips</div>
    <div class="tip-line"><span class="star" style="color:#4caf50;">★</span> <span style="color:#4caf50;">Success: Your tone Score is very stable...</span></div>
    <div class="tip-line"><span class="star" style="color:#4caf50;">★</span> <span style="color:#4caf50;">Warning Tips</span></div>
    <div class="tip-line"><span class="star" style="color:#f44336;">★</span> <span style="color:#ff9800;">Filler: Um and Ah detected...</span></div>
    <div class="tip-line"><span class="star" style="color:#00f2fe;">★</span> <span style="color:#00f2fe;">Info: Performance Score is high today</span></div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)