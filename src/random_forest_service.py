import mlflow
import mlflow.sklearn
from bentoml import BentoService, api, artifacts
from bentoml.sklearn import SklearnModelArtifact
from bentoml.adapters import JsonInput, JsonOutput
import pandas as pd

class RandomForestService(BentoService):
    
    @artifacts([SklearnModelArtifact('model')])
    def __init__(self, model):
        super().__init__()
        self.artifacts.model = model
    
    @api(input=JsonInput(), output=JsonOutput())
    def predict(self, json_data):
        df = pd.DataFrame(json_data)
        prediction = self.artifacts.model.predict(df)
        return prediction.tolist()

def load_model(run_id):
    try:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

if __name__ == "__main__":
    # Replace this with your actual run ID from MLflow
    run_id = "60ccff98b0a34914a22c82d23d12b1f8"
    
    # Load the model from MLflow
    model = load_model(run_id)
    
    # Create an instance of the BentoService
    svc = RandomForestService(model)
    
    # Save the BentoService for deployment
    svc.save()