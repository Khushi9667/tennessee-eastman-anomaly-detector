import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import dynamic save paths from config
from config import METRICS_SAVE_PATH, CM_PLOT_SAVE_PATH, TIMELINE_PLOT_SAVE_PATH, TARGET_FAULT

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set, prints metrics, 
    and saves the classification report (JSON) and confusion matrix plot (PNG).
    """
    print(f"--- Evaluating Model for Fault {TARGET_FAULT} ---")
    
    # Generate Predictions
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Get report as dictionary to save as JSON, and as string to print
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)
    class_report_str = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(class_report_str)
    
    # Save Metrics to JSON
    metrics = {
        "fault_id": TARGET_FAULT,
        "accuracy": accuracy,
        "classification_report": class_report_dict
    }
    
    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics successfully saved to {METRICS_SAVE_PATH}")
    
    # Plot and Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (0)', f'Fault {TARGET_FAULT} (1)'], 
                yticklabels=['Normal (0)', f'Fault {TARGET_FAULT} (1)'])
    
    plt.title(f"Confusion Matrix: Model Performance (Fault {TARGET_FAULT})")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    
    plt.tight_layout()
    plt.savefig(CM_PLOT_SAVE_PATH)
    plt.close() # Close the plot to free memory
    print(f"Confusion matrix plot successfully saved to {CM_PLOT_SAVE_PATH}")

    print(f"--- Generating Timeline Visuals for Fault {TARGET_FAULT} ---")
    """
    Evaluates the model and generates a continuous timeline plot showing 
    normal operation transitioning into a fault state.
    """
    
    # Extract exactly one Simulation Run (960 samples) from the test data
    # In TEP, the fault is always introduced at sample 160 of the faulty test run.
    X_run = X_test.iloc[-960:].copy()
    y_run = y_test.iloc[-960:].copy()
    
    # Dynamically find the most important feature to plot
    importances = model.feature_importances_
    top_feature_idx = importances.argmax()
    top_feature_name = X_test.columns[top_feature_idx]
    
    # Get model's probability predictions for this specific run
    probs = model.predict_proba(X_run)[:, 1]
    
    # Generate the Dual-Timeline Plot
    # Styled with a warm, neutral, atmospheric palette
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    
    # Top Plot: Sensor Drift
    ax1.plot(X_run[top_feature_name].values, color='#8b6d5c', linewidth=2, label=f'Sensor: {top_feature_name}')
    ax1.axvline(x=160, color='#c97a36', linestyle='--', linewidth=2, label='Fault Introduced')
    ax1.set_title(f"Sensor Behavior Over Time (Most Critical Feature: {top_feature_name})", color='#5c4d43', fontsize=14, fontweight='bold')
    ax1.set_ylabel("Standardized Value", color='#5c4d43')
    ax1.grid(True, linestyle=':', color='#e0d5c1')
    ax1.legend(loc="upper left")
    
    # Bottom Plot: Model Confidence Tracker
    ax2.plot(probs, color='#c97a36', linewidth=2, label='Model Confidence')
    ax2.fill_between(range(len(probs)), 0, probs, color='#d4af37', alpha=0.3)
    ax2.axvline(x=160, color='#8b6d5c', linestyle='--', linewidth=2)
    ax2.set_title("Machine Learning Anomaly Detection Status", color='#5c4d43', fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time (Samples)", color='#5c4d43', fontsize=12)
    ax2.set_ylabel("Fault Probability", color='#5c4d43')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', color='#e0d5c1')
    ax2.legend(loc="upper left")
    
    # Clean up formatting and save
    plt.tight_layout()
    plt.savefig(TIMELINE_PLOT_SAVE_PATH, dpi=150, bbox_inches='tight', transparent=True)
    plt.close() 
    
    print(f"Timeline plot successfully saved to {TIMELINE_PLOT_SAVE_PATH}")
    
    return metrics