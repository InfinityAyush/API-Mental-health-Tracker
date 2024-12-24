
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app= FastAPI()

class scorningItem(BaseModel):
    Q1A : int
    Q2A : int
    Q3A : int
    Q4A : int
    Q5A : int
    Q6A : int
    Q7A : int
    Q8A : int   
    Q9A : int
    Q10A : int
    Q11A : int
    Q12A : int

filename = 'best_svc_model.pkl'
with open(filename, 'rb') as file: # 'rb' stands for "read binary"
    model = pickle.load(file)
    

# @app.get('/')
# async def scoring_endpoint():
#     return {"hello":"World"}

@app.post('/')
async def scoring_endpoint(item:scorningItem):
    df = pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    yhat = model.predict(df)
    if yhat == 1:
        yhat = "you're good"
    elif yhat == 2:
         yhat = "Little bit streed"
    elif yhat == 3:
        yhat = "Sad"
    else:
         yhat = "You're deppresed"
    
    return {"Prediction": yhat}