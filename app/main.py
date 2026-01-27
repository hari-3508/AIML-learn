from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load("model/diabetes.pkl")

app = FastAPI()
@app.get("/")
def home():
    return{"message": "Diabetes predictor API is running"}

@app.post('/predict')

def predice_diabetes(data: dict):
    features = np.array([list(data.values())])
    prediction = model.predict(features)

    result = {
        "prediction": int(prediction[0])
    }
    
    return result