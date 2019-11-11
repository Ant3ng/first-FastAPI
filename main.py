from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import os
import base64
import io
import numpy as np
import torch
from torch import nn

app = FastAPI()


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def flatten(x):
    return x.view(len(x), -1)


def mnist_resize(x):
    return x.view(-1, 1, 28, 28)


def predict(img_path):
    # decode for loading img
    img = base64.b64decode(img_path.enc_img)
    # convert RGB into gray scale
    img = Image.open(io.BytesIO(img)).resize((28, 28)).convert("L")
    img = torch.tensor(np.array(img)).view(-1, 1, 28, 28).float()

    # load pre-trained model
    weight = Path(os.getcwd()) / "weight_cnn.pth"
    model = nn.Sequential(
        Lambda(mnist_resize),
        nn.Conv2d(1, 8, 5, 2, 2), nn.ReLU(),
        nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        nn.Linear(32, 10)
    )
    model.load_state_dict(torch.load(weight))

    # inference
    pred = model(img)
    pred = torch.argmax(pred, 1)

    return pred.item()


class Item(BaseModel):
    enc_img: bytes


@app.post("/pred/")
async def return_pred(file: Item):
    return predict(file)
