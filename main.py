import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post('/predict')
def predict_species(data:IrisSpecies):
    data = data.dict()
    loaded_model = pickle.load(open('RFclassifier.pkl', 'rb'))
    data_in = [[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]]
    prediction = loaded_model.predict(data_in)
    pred = prediction[0].item()
    if(pred == 0):
        species = 'Setosa'
    elif(pred == 1):
        species = 'Versicolor'
    else:
        species = 'Verginica'
    probability = loaded_model.predict_proba(data_in).max()
    return {
    'prediction': species,
    'probability': probability
    }

#if __name__ == '__main__':
#    uvicorn.run(app, host = '127.0.0.1', port = 8000)