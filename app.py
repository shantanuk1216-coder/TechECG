import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

st.set_page_config(page_title="ECG Dashboard", layout="wide")
st.title("🩺 Smart ECG Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your ECG CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    time_col, volt_col = df.columns[0], df.columns[1]
    time, raw_ecg = df[time_col].values, df[volt_col].values
    
    fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=500)
    
    b_notch, a_notch = iirnotch(50.0, 30.0, fs)
    ecg_notched = filtfilt(b_notch, a_notch, raw_ecg)
    b_band, a_band = butter(4, [0.5/(0.5*fs), 45.0/(0.5*fs)], btype='band')
    clean_ecg = filtfilt(b_band, a_band, ecg_notched)
    
    threshold = np.mean(clean_ecg) + 1.2 * np.std(clean_ecg)
    peaks, _ = find_peaks(clean_ecg, height=threshold, distance=int(0.4 * fs))
    
    bpm = 60 / np.mean(np.diff(peaks) / fs) if len(peaks) > 1 else 0
    st.success(f"Estimated Heart Rate: **{bpm:.1f} BPM**")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=clean_ecg, mode='lines', name='Filtered ECG', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=time[peaks], y=clean_ecg[peaks], mode='markers', name='R-Peaks', marker=dict(color='red', size=8)))
    fig.update_layout(title="Interactive ECG Waveform", xaxis_title="Time (s)", yaxis_title="Voltage (mV)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload a CSV file to begin.")