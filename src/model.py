import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from config import RF_PARAMS, MODEL_SAVE_PATH

def get_model_instance():
    """
    Returns an instance of the Random Forest model using hyperparameters 
    defined in config.py.
    """
    return RandomForestClassifier(**RF_PARAMS)

def train_model(X_train, y_train):
    """
    Initializes and trains the Random Forest model on the provided dataset.
    """
    print("--- Initializing and Training Random Forest Model ---")
    start_time = time.time()
    
    model = get_model_instance()
    model.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"Training Complete! Time taken: {(end_time - start_time):.2f} seconds.")
    
    return model

# Add this function at the bottom
def save_trained_model(model):
    """
    Saves the trained model to the directory specified in config.py.
    """
    print(f"--- Saving Model ---")
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model successfully saved to {MODEL_SAVE_PATH}")