from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from starlette.requests import Request
import requests, io
from pathlib import Path

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os

app = FastAPI()

def predict():
    # convert RGB into gray scale
    path = Path(os.getcwd())/"dataset/digit_2.png"
    img = Image.open(path).resize((28, 28)).convert("L")
    img = torch.tensor(np.array(img)).view(-1).float()

    # load pre-trained model
    weight = Path(os.getcwd())/"weight.pth"
    model  = nn.Sequential(nn.Linear(img.shape[-1], 50), nn.ReLU(), nn.Linear(50, 10))
    model.load_state_dict(torch.load(weight))

    # inference
    pred = model(img.view(1, -1))
    pred = torch.argmax(pred, 1)

    return pred.item()


@app.get("/predict/")
async def cast_predict():
    return {"pred": predict()}