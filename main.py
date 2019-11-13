from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from PIL import Image, ImageOps
import os
import base64
import io
import numpy as np
import torch
from models.spatial_tfm_nw import Stnet


app = FastAPI()


def normalize(x):
    mean, std = .1307, .3081
    return (x - mean) / std


def predict(img_path):
    # decode for loading img
    img = base64.b64decode(img_path.enc_img)
    # convert RGB into gray scale
    img = Image.open(io.BytesIO(img)).resize((28, 28)).convert("L")
    # pixel of input image is opposite to MNIST
    img = ImageOps.invert(img)

    # premise for spatial tfm nw is 4d tensor
    img = torch.tensor(np.array(img)).view(-1, 1, 28, 28).float()

    # Scale in training was btw 0 ~ 1, and normalized
    # So we need to rescale img, and then normalize
    img = normalize(img / 255)

    # load pre-trained model
    weight = Path(os.getcwd()) / "pre-trained/stn_30epoch.pt"
    model = Stnet()
    model.load_state_dict(torch.load(weight))

    # bs=1 prediction with BatchNorm should be in eval mode
    # Otherwise, you'll get an error
    # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
    model.eval()

    # inference
    pred = model(img)
    pred = torch.argmax(pred, 1)

    return pred.item()


class Item(BaseModel):
    enc_img: bytes


@app.post("/pred/")
async def return_pred(file: Item):
    return predict(file)
