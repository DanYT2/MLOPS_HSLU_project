from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

class CustomerData(BaseModel):
    age: int
    gender: str
    tenure: int
    monthly_charges: float
    total_charges: float
    


def predict(data: CustomerData):
    return {"prediction": "Hello, !"}

@app.post("/predict")
def predict_endpoint(req: CustomerData):
    prediction = predict(req)
    return {"prediction": prediction}