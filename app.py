import streamlit as st
import os
import time

# --- 1. PRECISE UI STYLING (Same as Reference Image) ---
st.set_page_config(page_title="AI Vocal Coach Pro", layout="wide")

st.markdown("""
    <style>
    /* Main Dark Theme */
    .stApp { background: #1a1c1e; color: #ffffff; }
    
    /* Performance Report Glass Box */
    .performance-container {
        background: #232629;
        border-radius: 15px;
        padding: 25px;
        border: 1px solid #363a3f;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        margin-top: 20px;
    }

    /* Top Audio Bar Style */
    .audio-player-mock {
        background: #232629;
        border-radius: 10px;
        padding: 10px;
        display: flex;
        align-items: center;
        border: 1px solid #363a3f;
        margin-bottom: 20px;
    }

    /* Metric Cards within the Report */
    .report-card {
        background: #2a2d32;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3c4148;
        height: 100%;
    }

    /* Start Recording Neon Button (The Red Glow) */
    div.stButton > button {
        background: transparent !important;
        color: #ffffff !important;
        border: 2px solid #ff4b4b !important;
        border-radius: 20px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        box-shadow: 0px 0px 10px rgba(255, 75, 75, 0.4);
        width: 100%;
        margin-top: 10px;
    }

    /* Coaching Tips Section Styling */
    .tips-box {
        background: rgba(0, 242, 254, 0.05);
        border: 1px solid rgba(0, 242, 254, 0.2);
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    
    .tip-item { display: flex; align-items: center; margin-bottom: 8px; font-size: 14px; }
    .star-icon { margin-right: 10px; font-size: 18px; }
    
    /* Metrics Typography */
    .metric-value { font-size: 48px; font-weight: 700; color: #ffffff; line-height: 1; }
    .metric-label { font-size: 14px; color: #8e949a; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LAYOUT CONSTRUCTION ---

# Top Audio Bar Mockup (Image reference: image_30f755.jpg)
st.caption("Recording Time (seconds)")
st.markdown("""
    <div class="audio-player-mock">
        <span style="margin-right:15px;">▶ 0:00 / 0:05</span>
        <div style="flex-grow:1; height:2px; background:#ff4b4b; position:relative;">
            <div style="width:10px; height:10px; background:#ff4b4b; border-radius:50%; position:absolute; right:0; top:-4px; box-shadow:0 0 10px #ff4b4b;"></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main Performance Report Section (Image reference: image_302c23.jpg)
st.markdown('<div class="performance-container">', unsafe_allow_html=True)
st.markdown('<p style="font-weight:600; font-size:18px; margin-bottom:15px;">📊 Performance Report</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">19</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Confidence Score</div>', unsafe_allow_html=True)
    if st.button("🔴 Start Recording"):
        st.toast("Recording started...")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="report-card" style="background:#1a1c1e;">', unsafe_allow_html=True)
    st.write("##")
    st.info("🔊 speech.wav")
    st.caption("transbich.wav ready")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">3%</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Fillers Found</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8e949a; font-size:12px; margin-top:10px;">Speaking Speed</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Coaching Tips Section (Image reference: image_30fa7e.jpg)
st.write("##")
st.write("What you said:")
st.markdown("""
    <div class="tips-box">
        <p style="font-size:16px; font-weight:600; color:#00f2fe;">💡 Coaching Tips</p>
        <div class="tip-item"><span class="star-icon">⭐</span> <span style="color:#4caf50;">Success Tips: Your tone is very stable today.</span></div>
        <div class="tip-item"><span class="star-icon" style="color:#4caf50;">⭐</span> <span style="color:#4caf50;">Success Time: High score in clarity.</span></div>
        <div class="tip-item"><span class="star-icon" style="color:#ff9800;">⭐</span> <span style="color:#ff9800;">Warning Tips: Slow down during intro.</span></div>
        <div class="tip-item"><span class="star-icon" style="color:#f44336;">⭐</span> <span style="color:#f44336;">Speaking Speed: Too fast at 0:03.</span></div>
    </div>
""", unsafe_allow_html=True)