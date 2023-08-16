import pickle
import requests
from code import predictcar
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can change this to a list of allowed origins
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    img: str 

MODEL_PATH =  'D:\AI\Assignment 1\CarBrandClass\model\imageCAR_model.pkl'
# MODEL_PATH = '/model\imageCAR_model.pkl'
HOG_API_URL = 'http://localhost:8080/api/gethog'
# 172.17.0.3:8080
headers = {"Content-Type": "application/json"}
# m = pickle.load(open(MODEL_PATH, 'rb'))

m = pickle.load(open(MODEL_PATH, 'rb'))

@app.get("/")
def root():
    return {"message": "This is my api imageCAR"}    

@app.post("/api/carbrand")
async def create_item(item: Item):
    jsons = {"img":item.img}
    response = requests.get(HOG_API_URL, json=jsons, headers=headers)
    hog = response.json()
    res = predictcar(m,[hog['Hog']])
    return {"predict":res}