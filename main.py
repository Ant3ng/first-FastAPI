from fastapi import FastAPI
from pydantic import BaseModel

from pathlib import Path
from PIL import Image
import os, base64, io

import numpy as np
import torch
from torch import nn


app = FastAPI()

def predict(img_path):
    # decode for loading img
    img = base64.b64decode(img_path.enc_img)
    # convert RGB into gray scale
    img = Image.open(io.BytesIO(img)).resize((28, 28)).convert("L")
    img = torch.tensor(np.array(img)).view(-1).float()

    # load pre-trained model
    weight = Path(os.getcwd())/"weight.pth"
    model  = nn.Sequential(nn.Linear(img.shape[-1], 50), nn.ReLU(), nn.Linear(50, 10))
    model.load_state_dict(torch.load(weight))

    # inference
    pred = model(img.view(1, -1))
    pred = torch.argmax(pred, 1)

    return pred.item()


class Item(BaseModel):
    enc_img: bytes

@app.post("/pred/")
async def return_pred(file: Item):
    return predict(file)

