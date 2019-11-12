from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import os
import base64
import io
import numpy as np
import torch
from fastai_resnet import resnet_model

app = FastAPI()


def predict(img_path):
    # decode for loading img
    img = base64.b64decode(img_path.enc_img)
    # convert RGB into gray scale
    img = Image.open(io.BytesIO(img)).resize((28, 28)).convert("L")
    img = torch.tensor(np.array(img)).view(-1, 1, 28, 28).float()

    # load pre-trained model
    weight = Path(os.getcwd()) / "fastai_resnet.pt"
    model = resnet_model()
    model.load_state_dict(torch.load(weight))

    # bs=1 prediction with BatchNorm should be in eval mode
    # Otherwise, you'll encounter error
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
