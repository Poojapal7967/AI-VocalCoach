import streamlit as st

# --- 1. ULTRA-NEON GLOW & GLASSMORPISM (Exact Match image_316f3a.png) ---
st.set_page_config(page_title="AI Vocal Coach Pro", layout="wide")

st.markdown("""
    <style>
    /* Dark Deep Theme */
    .stApp { background: #0c0c1e; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    /* PERFORMANCE REPORT Header with Icon */
    .header-style {
        color: #ffcc00; font-weight: 700; font-size: 18px;
        display: flex; align-items: center; margin-bottom: 20px;
        text-transform: uppercase; letter-spacing: 1px;
    }

    /* TOP AUDIO BAR with Neon Red Glow (image_316f3a) */
    .audio-player {
        background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 12px 20px; margin-bottom: 30px;
        display: flex; align-items: center; justify-content: space-between;
        backdrop-filter: blur(10px);
    }
    .glow-line { flex-grow: 1; height: 3px; background: #ff4b4b; margin: 0 20px; position: relative; border-radius: 5px; }
    .glow-dot { 
        width: 14px; height: 14px; background: #ff4b4b; border-radius: 50%; 
        position: absolute; right: 0; top: -5px; 
        box-shadow: 0 0 15px 5px rgba(255, 75, 75, 0.8); 
    }

    /* NEON GLOWING CARDS (Precise Border Glow from image_316f3a) */
    .neon-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px; padding: 50px 20px; text-align: center;
        backdrop-filter: blur(15px); height: 280px;
        display: flex; flex-direction: column; justify-content: center;
        transition: 0.4s; position: relative;
    }

    /* Red Glow Card */
    .card-red { 
        border: 2px solid #ff4b4b; 
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.4); 
        filter: drop-shadow(0 0 8px rgba(255, 75, 75, 0.3));
    }
    /* Orange Glow Card */
    .card-orange { 
        border: 2px solid #ff9800; 
        box-shadow: 0 0 20px rgba(255, 152, 0, 0.4); 
        filter: drop-shadow(0 0 8px rgba(255, 152, 0, 0.3));
    }
    /* Blue Glow Card */
    .card-blue { 
        border: 2px solid #00f2fe; 
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.4); 
        filter: drop-shadow(0 0 8px rgba(0, 242, 254, 0.3));
    }

    /* Typography inside Cards */
    .big-value { font-size: 75px; font-weight: 800; color: #ffffff; line-height: 1; margin-bottom: 5px; }
    .label-sub { font-size: 14px; color: #8e949a; text-transform: uppercase; letter-spacing: 2px; font-weight: 500; }

    /* START RECORDING Button with Heavy Neon Glow */
    div.stButton > button {
        background: transparent !important; color: #ffffff !important;
        border: 2px solid #ff4b4b !important; border-radius: 40px !important;
        padding: 15px 40px !important; font-weight: 700 !important;
        box-shadow: 0px 0px 25px rgba(255, 75, 75, 0.5); 
        width: 300px !important; margin-top: 30px; transition: 0.3s;
        text-transform: uppercase;
    }
    div.stButton > button:hover { background: #ff4b4b !important; box-shadow: 0px 0px 40px #ff4b4b; color: #000 !important; }

    /* Center WAV Box */
    .inner-wav { background: rgba(0,0,0,0.3); border-radius: 12px; padding: 18px; border: 1px solid #333; width: 90%; margin: 0 auto; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. TOP AUDIO PLAYER BAR ---
st.caption("Recording Time (seconds)")
st.markdown("""
    <div class="audio-player">
        <span style="font-weight:600;">▶ 0:00 / 0:05</span>
        <div class="glow-line"><div class="glow-dot"></div></div>
        <span style="font-size:20px;">🔊 ⋮</span>
    </div>
""", unsafe_allow_html=True)

# --- 3. PERFORMANCE GRID WITH NEON EFFECTS ---
st.markdown('<div class="header-style">📊 PERFORMANCE REPORT</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
        <div class="neon-card card-red">
            <div class="big-value">19</div>
            <div class="label-sub">Confidence Score</div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
        <div class="neon-card card-orange">
            <div class="inner-wav">🔊 speech.wav</div>
            <div class="label-sub" style="margin-top:25px;">Transbich.wav Ready</div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
        <div class="neon-card card-blue">
            <div class="big-num" style="font-size:75px; font-weight:800;">3%</div>
            <div class="label-sub">Fillers Found</div>
            <div style="font-size:12px; color:#8e949a; margin-top:20px; font-style:italic;">Speaking: Normal</div>
        </div>
    """, unsafe_allow_html=True)

# START RECORDING Button at the bottom
st.write("##")
if st.button("🔴 START RECORDING"):
    st.toast("Neon AI Engine Active...")

# --- 4. COACHING FEEDBACK (image_316f3a style) ---
st.write("##")
st.markdown("""
    <div style="background: rgba(0, 242, 254, 0.05); border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 15px; padding: 25px;">
        <p style="color:#00f2fe; font-weight:bold; font-size:18px; margin-bottom:20px;">💡 COACHING TIPS</p>
        <div style="margin-bottom:12px; display:flex; align-items:center;">
            <span style="color:#4caf50; font-size:20px; margin-right:15px;">★</span> 
            <span style="color:#4caf50; font-weight:500;">Success: Your vocal tone is very stable today.</span>
        </div>
        <div style="margin-bottom:12px; display:flex; align-items:center;">
            <span style="color:#ff9800; font-size:20px; margin-right:15px;">★</span> 
            <span style="color:#ff9800; font-weight:500;">Warning: Detected slight filler word frequency at 0:03.</span>
        </div>
    </div>
""", unsafe_allow_html=True)