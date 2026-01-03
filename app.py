import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa

# AI Model Loading
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

st.title("🎙️ AI Vocal Coach Pro")
st.write("Improve your Public Speaking with Real-time AI Feedback")

# Recording Duration
duration = st.slider("Select Recording Duration (seconds)", 3, 15, 5)

if st.button("🔴 Start Recording"):
    fs = 44100
    with st.spinner("Listening... Speak now!"):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write('speech.wav', fs, recording)
    
    st.success("Analysis Complete!")
    st.audio('speech.wav')

    # AI ANALYSIS SECTION
    with st.spinner("AI is analyzing your voice..."):
        # 1. Transcription
        result = model.transcribe("speech.wav")
        text = result['text']
        
        # 2. Pitch & Tone Analysis
        y, sr = librosa.load("speech.wav")
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    # DISPLAY RESULTS
    st.subheader("📊 Performance Report")
    st.write(f"**Transcribed Text:** {text}")
    
    # Confidence Score Calculation (Basic Logic)
    fillers = ["um", "uh", "ah", "like", "hmm"]
    filler_count = sum(1 for word in text.lower().split() if word in fillers)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Filler Words", filler_count)
    col2.metric("Pitch (Hz)", f"{int(avg_pitch)}")
    col3.metric("Speech Rate", "Normal" if len(text.split()) > 5 else "Slow")

    # Final Advice
    st.info("💡 **Expert Advice:**")
    if filler_count > 0:
        st.write("- Try to reduce 'um' and 'uh' by taking small pauses instead.")
    if avg_pitch < 120:
        st.write("- Your voice is a bit flat. Try to add more energy and modulation.")