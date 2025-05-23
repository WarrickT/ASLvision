# from model import train_model

# train_model.run()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.predict import predict_from_landmarks
from typing import List

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aslvision-frontend.onrender.com",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# This defines the shape of the incoming request body
# FastAPI parses the JSON and checks for a "landmarks" field that is a list of floats
class LandmarksRequest(BaseModel):
    landmarks: List[float]

# predict endpoint

@app.post("/predict")
def predict_letter(data: LandmarksRequest):
    if(len(data.landmarks) != 63):
        raise HTTPException(status_code=400, detail="Expected 63 landmark values!")
    
    letter = predict_from_landmarks(data.landmarks)
    return {"letter": letter}