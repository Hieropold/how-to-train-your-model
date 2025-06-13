from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model using joblib
model = joblib.load("hint_lover_model.pkl")

class ModelInput(BaseModel):
  feature names...

# Initialize FastAPI app
app = FastAPI()


@app.post("/predict")
def predict(input_data: ModelInput):
    # Prepare input data for prediction
    data = np.array([[input_data.feature1, input_data.feature2, input_data.feature3]])

    # Make prediction
    prediction = model.predict(data)

    # Return the result as a float
    return {"prediction": float(prediction[0])}