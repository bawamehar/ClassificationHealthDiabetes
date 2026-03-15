import pandas as pd
from fastapi import FastAPI 
from pydantic import BaseModel
import psycopg2
from .patientService import insert_patient


app= FastAPI()

# catch streamlit data
class request_body(BaseModel):
    bmi : float
    age : float
    genhlth : float
    income : float
    highbp : float
    highchol : float
    education : float

@app.get("/")
async def root():
    return {"Welcome Diabetes Prediction"}

@app.post("/predict")
async def predict(stlmdata: request_body):
    print("f1")
    formData = stlmdata.dict()
    values = tuple(formData.values())   # Convert dict to tuple
    print("f2")
    insert_patient(values)
    return {"message": "Patient inserted successfully", "data": formData}