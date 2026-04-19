import streamlit as st
import json
import os
import pandas as pd
from PIL import Image
import subprocess
import sys

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="TEP Anomaly Detection",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS for a warm, neutral aesthetic 
st.markdown("""
<style>
    .stApp {
        background-color: #fcfaf8;
    }
    .stButton > button {
        background-color: #c97a36; /* Burnt orange */
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #a85e24;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        color: #5c4d43; /* Deep warm brown */
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# PATH DEFINITIONS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_PATH = os.path.join(BASE_DIR, "src", "config.py")
MAIN_SCRIPT_PATH = os.path.join(BASE_DIR, "src", "main.py")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def update_target_fault_in_config(fault_id):
    """
    Dynamically updates the TARGET_FAULT variable in the config.py file 
    so the backend pipeline knows which fault to process.
    """
    with open(CONFIG_PATH, "r") as f:
        lines = f.readlines()
    
    with open(CONFIG_PATH, "w") as f:
        for line in lines:
            if line.startswith("TARGET_FAULT ="):
                f.write(f"TARGET_FAULT = {fault_id}\n")
            else:
                f.write(line)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚙️ TEP Dashboard")
st.sidebar.markdown("Select a fault to view its analysis or train a new model.")

selected_fault = st.sidebar.selectbox(
    "Target Fault ID:",
    options=list(range(1, 21)),
    index=19 # Defaults to 20
)

# ==========================================
# MAIN DASHBOARD AREA
# ==========================================
st.title("🏭 Tennessee Eastman Process Anomaly Detector")
st.markdown(f"### Evaluating Performance for **Fault {selected_fault}**")

metrics_file = os.path.join(REPORTS_DIR, f"metrics_fault_{selected_fault}.json")
cm_image_file = os.path.join(REPORTS_DIR, f"confusion_matrix_fault_{selected_fault}.png")

# Check if the pipeline has already been run for this fault
if os.path.exists(metrics_file) and os.path.exists(cm_image_file):
    
    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)
    
    st.markdown("#### 🎯 Quick Stats")
    col1, col2, col3 = st.columns(3)
    
    accuracy = metrics_data.get("accuracy", 0)
    col1.metric("Overall Accuracy", f"{accuracy * 100:.2f}%")
    
    report = metrics_data.get("classification_report", {})
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0)
    col2.metric("Macro F1-Score", f"{macro_f1:.2f}")
    
    col3.metric("Model Used", "Random Forest")
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 📊 Confusion Matrix")
        image = Image.open(cm_image_file)
        st.image(image, use_container_width=True)
        
    with col_right:
        st.markdown("#### 📑 Classification Report")
        report_df = pd.DataFrame(report).transpose()
        if 'accuracy' in report_df.index:
            report_df = report_df.drop('accuracy')
            
        st.dataframe(
            report_df.style.background_gradient(cmap='Oranges'),
            use_container_width=True,
            height=300
        )

else:
    # --- DYNAMIC TRAINING UI ---
    st.info(f"No trained model found for **Fault {selected_fault}**. You can generate it right here.")
    
    if st.button(f"⚙️ Run ML Pipeline for Fault {selected_fault}"):
        
        # 1. Update the config file to point to the selected fault
        update_target_fault_in_config(selected_fault)
        
        # UI Elements for Progress Tracking
        progress_bar = st.progress(0)
        status_text = st.empty() # Holds the current active step
        log_window = st.empty()  # Holds the streaming terminal logs
        
        status_text.markdown("**Status:** Initializing pipeline...")
        
        # 2. Trigger the pipeline script asynchronously using Popen
        # We pass PYTHONUNBUFFERED=1 to ensure prints flow immediately to the UI
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        process = subprocess.Popen(
            [sys.executable, MAIN_SCRIPT_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Route errors to the same output
            text=True,
            bufsize=1,
            env=env
        )
        
        logs = []
        
        # 3. Read the terminal output line-by-line as it runs
        for line in iter(process.stdout.readline, ''):
            clean_line = line.strip()
            if not clean_line:
                continue
                
            # Append to our running log list
            logs.append(clean_line)
            
            # Keep only the last 15 lines so the box doesn't grow infinitely huge
            display_logs = "\n".join(logs[-15:])
            
            # Render the logs in a small-font, terminal-style box
            log_window.markdown(
                f"""
                <div style="background-color: #f4efe9; padding: 10px; border-radius: 5px; 
                            border: 1px solid #e0d5c1; font-family: monospace; 
                            font-size: 11px; color: #5c4d43; white-space: pre-wrap;">
{display_logs}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # 4. Dynamically update the progress bar based on what the script prints
            if "Fetching Data for Fault" in clean_line:
                progress_bar.progress(10)
                status_text.markdown("**Status:** Loading massive datasets into memory...")
            elif "Starting Preprocessing" in clean_line:
                progress_bar.progress(30)
                status_text.markdown("**Status:** Capping outliers and scaling features...")
            elif "Calculating feature importance" in clean_line:
                progress_bar.progress(50)
                status_text.markdown("**Status:** Selecting top features and checking redundancy...")
            elif "Initializing and Training Random Forest" in clean_line:
                progress_bar.progress(70)
                status_text.markdown("**Status:** Training Random Forest model...")
            elif "Evaluating Model" in clean_line:
                progress_bar.progress(85)
                status_text.markdown("**Status:** Generating confusion matrix and metrics...")
            elif "PIPELINE COMPLETE" in clean_line:
                progress_bar.progress(100)
                status_text.markdown("**Status:** Done!")

        # Wait for the process to fully close
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            st.success("Pipeline completed successfully! Refreshing dashboard...")
            st.rerun()
        else:
            st.error("❌ An error occurred during training. Please check the logs above.")