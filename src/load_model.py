import mlflow
import mlflow.sklearn

run_id = "60ccff98b0a34914a22c82d23d12b1f8"
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
print(model)