# bring lightweight libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import sklearn

app = FastAPI()

class Scoringitem(BaseModel):
    """Scoring item class to be used for the API endpoint"""
    YearsAtCompany: int # 1, // Float value
    EmployeeSatisfaction: float #0.01, // Float value
    Position: str #"Non-Manager", // Manager or Non-Manager
    Salary: int # 4.0 // Ordinal 1,2,3,4,5p


with open('rfmodel.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/")
async def scoring_endpoint(item: Scoringitem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df) 
    return {"prediction": int(yhat)}