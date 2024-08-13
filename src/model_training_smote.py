import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_and_process_data
from sklearn.metrics import accuracy_score, classification_report
from mlflow.models import infer_signature
from imblearn.over_sampling import SMOTE

def train_and_log_models_with_smote():
    # Load and process data
    X_train, X_test, y_train, y_test = load_and_process_data('data/predictive_maintenance.csv')
    
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    
    for model_name, model in models.items():
        with mlflow.start_run():
            # Train model
            model.fit(X_train_resampled, y_train_resampled)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Log parameters and metrics
            mlflow.log_param("model", model_name)
            mlflow.log_param("train_size", len(X_train_resampled))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            
            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")
            
            # Infer model signature
            signature = infer_signature(X_train_resampled, model.predict(X_train_resampled))
            
            # Log the model with signature
            mlflow.sklearn.log_model(model, "model", signature=signature)
            
            print(f"Logged {model_name} with SMOTE with accuracy: {accuracy}")

if __name__ == "__main__":
    train_and_log_models_with_smote()