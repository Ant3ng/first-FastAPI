from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from PIL import Image, ImageOps
import os
import base64
import io
import numpy as np
import torch
import torch.nn.functional as F
from models.spatial_tfm_nw import Stnet

app = FastAPI()


def normalize(x):
    mean, std = .1307, .3081
    return (x - mean) / std


def preprocess_img(img_item):
    # decode for loading img
    img = base64.b64decode(img_item.enc_img)
    # convert RGB into gray scale
    img = Image.open(io.BytesIO(img)).resize((28, 28)).convert("L")
    # pixel of input image is opposite to MNIST
    img = ImageOps.invert(img)

    # premise for spatial tfm nw is 4d tensor
    img = torch.tensor(np.array(img)).view(-1, 1, 28, 28).float()

    # Scale in training was btw 0 ~ 1, and normalized
    # So we need to rescale img, and then normalize
    img = normalize(img / 255)

    return img


def check_confidence(conf, threshold):
    if conf > threshold:
        return "Yes"
    return "No"


def predict(img_item):
    img = preprocess_img(img_item)

    # load pre-trained model
    weight = Path(os.getcwd()) / "pre-trained/stn_30epoch.pt"
    model = Stnet()
    model.load_state_dict(torch.load(weight))

    # bs=1 prediction with BatchNorm should be in eval mode
    # Otherwise, you'll get an error
    # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
    model.eval()

    # inference
    pred_num = model(img)

    # How degree this model is confident to pred
    # prob = (torch.exp(pred) / torch.exp(pred).sum()).tolist()[0]
    prob = F.softmax(torch.tensor(pred_num).float()).tolist()[0]

    # score order which this model makes
    pred_order = torch.argsort(torch.tensor(prob), descending=True).tolist()

    # compare which is higher between pred and threshold
    check_conf = check_confidence(prob[pred_order[0]], img_item.threshold)

    # To make prob more intuitive
    prob = (np.round(np.array(prob, float), 6))
    prob = {i: prob[i] for i in range(10)}

    return {'pred_num': pred_order[0],
            'pred_order': pred_order,
            'prob': prob,
            'threshold': img_item.threshold,
            'check_conf': check_conf}


class Item(BaseModel):
    enc_img: bytes
    threshold: float


@app.post("/pred/")
async def return_pred(file: Item):
    return predict(file)
