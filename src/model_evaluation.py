import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from data_preprocessing import load_and_process_data
from model_training import train_and_log_models
 
def evaluate_model(model_uri, X_test, y_test):
    # Load the model from MLflow
    model = mlflow.sklearn.load_model(model_uri)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print results
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_process_data('data/predictive_maintenance.csv')
    
    # Train and log models
    train_and_log_models()
    
    # Evaluate models (example using the first model)
    model_uri = "runs:/f50954ff062e40b8a69539eee946a4d8/model"  # Replace <run_id> with your actual run ID
    evaluate_model(model_uri, X_test, y_test)
