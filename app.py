import streamlit as st
import json
import os
import pandas as pd
from PIL import Image
import subprocess
import sys

# PAGE CONFIGURATION
st.set_page_config(
    page_title="TEP Anomaly Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ADVANCED FRONTEND STYLING (CSS)
st.markdown("""
<style>
    /* Typography Adjustments */
    h1, h2, h3, h4 {
        color: var(--primary-color) !important;
        font-family: 'Georgia', serif;
        padding-bottom: 0.5rem;
    }
    
    p {
        font-size: 1.05rem;
        color: var(--text-color);
    }

    /* Custom Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 2rem;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: var(--secondary-background-color);
        border-top: 4px solid var(--primary-color);
        border-radius: 8px;
        padding: 24px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        flex: 1;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-color);
    }

    /* Plot Image Container */
    .plot-container {
        background-color: var(--background-color);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--secondary-background-color);
    }
    
    /* Custom Button */
    .stButton > button {
        background-color: var(--primary-color); 
        color: brown;
        border-radius: 6px;
        border: none;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        opacity: 0.85;
        color: brown;
        border: 1px solid var(--primary-color);
    }
            
    /* Project Description Box */
    .info-box {
        background-color: var(--secondary-background-color);
        border-left: 4px solid var(--primary-color);
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 25px;
        margin-top: 5px;
        color: var(--text-color);
        font-size: 1rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .info-box strong {
        color: var(--primary-color);
        font-family: 'Georgia', serif;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# PATH DEFINITIONS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_PATH = os.path.join(BASE_DIR, "src", "config.py")
MAIN_SCRIPT_PATH = os.path.join(BASE_DIR, "src", "main.py")

# HELPER FUNCTIONS
def update_target_fault_in_config(fault_id):
    with open(CONFIG_PATH, "r") as f:
        lines = f.readlines()
    with open(CONFIG_PATH, "w") as f:
        for line in lines:
            if line.startswith("TARGET_FAULT ="):
                f.write(f"TARGET_FAULT = {fault_id}\n")
            else:
                f.write(line)

# TEP FAULT DESCRIPTIONS
# Standard definitions for the Tennessee Eastman Process benchmark
FAULT_DESCRIPTIONS = {
    1: "A/C Feed Ratio, B Composition Constant (Step variable)",
    2: "B Composition, A/C Ratio Constant (Step variable)",
    3: "D Feed Temperature (Step variable)",
    4: "Reactor Cooling Water Inlet Temperature (Step variable)",
    5: "Condenser Cooling Water Inlet Temperature (Step variable)",
    6: "A Feed Loss (Step variable)",
    7: "C Header Pressure Loss - Reduced Availability (Step variable)",
    8: "A, B, C Feed Composition (Random variation)",
    9: "D Feed Temperature (Random variation)",
    10: "C Feed Temperature (Random variation)",
    11: "Reactor Cooling Water Inlet Temperature (Random variation)",
    12: "Condenser Cooling Water Inlet Temperature (Random variation)",
    13: "Reaction Kinetics (Slow drift)",
    14: "Reactor Cooling Water Valve (Sticking)",
    15: "Condenser Cooling Water Valve (Sticking)",
    16: "Unknown Process Disturbance",
    17: "Unknown Process Disturbance",
    18: "Unknown Process Disturbance",
    19: "Unknown Process Disturbance",
    20: "Unknown Process Disturbance"
}

# SIDEBAR
st.sidebar.markdown("## Dashboard Controls")
st.sidebar.markdown("Select a fault parameters to view system analysis.")

selected_fault = st.sidebar.selectbox(
    "Target Fault ID",
    options=list(range(1, 21)),
    index=19
)

st.sidebar.markdown("---")
st.sidebar.caption("Tennessee Eastman Process Anomaly Detection System v1.0")

# MAIN DASHBOARD AREA
st.title("Plant Health & Anomaly Detector")

# Grab the physical description of the selected fault
fault_context = FAULT_DESCRIPTIONS.get(selected_fault, "Unknown Disturbance")

st.markdown(f"""
<div class="info-box">
    The Tennessee Eastman Process (TEP) is a rigorous benchmark chemical simulation used to evaluate plant control and anomaly detection systems. This dashboard operates on top of a dynamic Machine Learning pipeline. It monitors continuous process telemetry, automatically isolates critical sensor features, and deploys a Random Forest algorithm to detect and classify 20 distinct fault conditions—visualizing sensor drift and confidence probability in real-time.
    <br><br>
    <span style="color: var(--primary-color); font-weight: bold;">Current Target (Fault {selected_fault}):</span> {fault_context}
</div>
""", unsafe_allow_html=True)

st.markdown(f"**Diagnostic Overview:** Monitoring system metrics for **Fault {selected_fault}**.")

metrics_file = os.path.join(REPORTS_DIR, f"metrics_fault_{selected_fault}.json")
timeline_image_file = os.path.join(REPORTS_DIR, f"timeline_fault_{selected_fault}.png")

if os.path.exists(metrics_file) and os.path.exists(timeline_image_file):
    
    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)
    
    # --- CUSTOM METRIC CARDS ---
    accuracy = metrics_data.get("accuracy", 0) * 100
    
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-title">Overall Accuracy</div>
                <div class="metric-value">{accuracy:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Pipeline Status</div>
                <div class="metric-value">Active</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Primary Architecture</div>
                <div class="metric-value">Random Forest</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # --- PLOT SECTION ---
    st.markdown("### System Health Timeline")
    st.markdown(
        "Observation of a single continuous simulation run. "
        "The system operates in a healthy state for the first 160 samples before the fault injection occurs, triggering the confidence tracker."
    )
    
    # Wrap the image in a styled container
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    image = Image.open(timeline_image_file)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # --- DYNAMIC TRAINING UI ---
    st.info(f"System requires initialization for **Fault {selected_fault}**.")
    
    if st.button(f"Run Diagnostic Pipeline (Fault {selected_fault})"):
        update_target_fault_in_config(selected_fault)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown("**Status:** Initializing pipeline...")
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        process = subprocess.Popen(
            [sys.executable, MAIN_SCRIPT_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            env=env
        )
        
        # Read the terminal output silently to update the progress bar
        for line in iter(process.stdout.readline, ''):
            clean_line = line.strip()
            if not clean_line: continue
                
            if "Fetching Data for Fault" in clean_line:
                progress_bar.progress(10)
                status_text.markdown("**Status:** Extracting process telemetry...")
            elif "Starting Preprocessing" in clean_line:
                progress_bar.progress(30)
                status_text.markdown("**Status:** Standardizing signals and capping noise...")
            elif "Calculating feature importance" in clean_line:
                progress_bar.progress(50)
                status_text.markdown("**Status:** Isolating critical sensors...")
            elif "Initializing and Training Random Forest" in clean_line:
                progress_bar.progress(70)
                status_text.markdown("**Status:** Training Random Forest architecture...")
            elif "Generating Timeline" in clean_line:
                progress_bar.progress(85)
                status_text.markdown("**Status:** Generating behavioral timeline...")
            elif "PIPELINE COMPLETE" in clean_line:
                progress_bar.progress(100)
                status_text.markdown("**Status:** Compilation complete.")

        process.stdout.close()
        if process.wait() == 0:
            st.success("Diagnostic pipeline initialized successfully. Refreshing view...")
            st.rerun()
        else:
            st.error("Process failure. Retry later.")