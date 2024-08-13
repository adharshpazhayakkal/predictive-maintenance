import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_process_data

def load_and_test_models():
    # Load and process data
    _, X_test, _, y_test = load_and_process_data('data/predictive_maintenance.csv')

    # Dictionary of model names and their corresponding run IDs
    model_run_ids = {
        "Logistic Regression": "75bb84c9e25848e3a3c036f5d5ab2347",  # Replace with actual run ID
        "Decision Tree": "5e713a593dae47f7b182adee0856460a",  # Replace with actual run ID
        "Random Forest": "60ccff98b0a34914a22c82d23d12b1f8"  # Replace with actual run ID
    }

    # Load and evaluate each model
    for model_name, run_id in model_run_ids.items():
        # Construct the model URI
        model_uri = f"runs:/{run_id}/model"
        
        # Load the model
        model = mlflow.sklearn.load_model(model_uri)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Print classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        print("Classification Report:")
        for label, metrics in report.items():
            if isinstance(metrics, dict):  # Check if metrics is a dictionary
                print(f"{label}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            else:  # Handle accuracy (which is a float)
                print(f"{label}: {metrics:.4f}")
        print("\n")

if __name__ == "__main__":
    load_and_test_models()