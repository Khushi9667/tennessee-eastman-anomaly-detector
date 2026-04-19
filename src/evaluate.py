import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import dynamic save paths from config
from config import METRICS_SAVE_PATH, CM_PLOT_SAVE_PATH, TARGET_FAULT

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
    
    return metrics