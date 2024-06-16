import pandas as pd
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from dill import load
from pathlib import Path
from fastapi import FastAPI, status
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import __main__
import uvicorn
import logging
from os import environ

# Fix pandas not found https://stackoverflow.com/a/65318623
__main__.pandas = pd

MODE = environ.get("MODE", "dev")
logger = logging.getLogger("uvicorn.error")

data = pd.read_csv("./datasets/players_22.csv", low_memory=False)
y = data[["overall"]]

modelPath = Path("./server/Fifa_Model.pkl")


def loadModel(modelPath: Path):
    model = load(open(modelPath, mode="rb"))
    return model


def r5(x):
    return np.round(x, 5)


def modelMetrics(model, preds, actuals) -> str:
    rmse = r5(root_mean_squared_error(preds, actuals))
    mae = r5(mean_absolute_error(preds, actuals))
    r2s = r5(r2_score(preds, actuals))
    score = model.score(data, actuals)
    return f"RMSE: {rmse} MAE: {mae} R2S: {r2s} Accuracy: {score:0.2%}"


class DataPoint(BaseModel):
    movement_reactions: float
    mentality_composure: float
    wage_eur: float
    release_clause_eur: float
    value_eur: float
    age: int
    physic: float
    pace: float
    shooting: float
    passing: float
    dribbling: float
    defending: float


origins = ["http://localhost:8080", "http://127.0.0.1:53400", "http://127.0.0.1:8080"]
model = loadModel(modelPath)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(status_code=status.HTTP_200_OK):
    preds = model.predict(data)
    res = modelMetrics(model, preds, y)
    logger.info(f"Health: {res}")
    return {"data": "Up"}


@app.post("/predict/")
def predict(dataPoint: DataPoint):
    data = dataPoint.model_dump()
    data = pd.DataFrame(data, index=[0])
    prediction = model.predict(data)
    logger.info(f"Prediction: {prediction[0]}")
    predictionFmted = np.floor(prediction[0])
    return {"data": predictionFmted}


if __name__ == "__main__":
    host = "127.0.0.1" if MODE == "dev" else "0.0.0.0"
    uvicorn.run(app, host=host, port=8000)
