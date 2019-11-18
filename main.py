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
import json


app = FastAPI()


def normalize(x):
    mean, std = .1307, .3081
    return (x - mean) / std


def preprocess_img(img_path):
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

    return img


def check_confidence(conf, threshold):
    if conf > threshold:
        conf = np.round(np.array(conf, np.float32), 6)
        print("This model's confidence ({}) is higher than threshold ({})".format(conf, threshold))
        return "Yes"
    print("This model's confidence ({}) is lower than threshold ({})".format(conf, threshold))
    return "No"


def predict(img_path):

    threshold = .50
    # threshold = .25
    # threshold = .10
    # threshold = .05

    img = preprocess_img(img_path)

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

    # How degree this model is confident to pred
    # prob = (torch.exp(pred) / torch.exp(pred).sum()).tolist()[0]
    prob = F.softmax(torch.tensor(pred).float()).tolist()[0]

    # score order which this model makes
    pred_order = torch.argsort(torch.tensor(prob), descending=True).tolist()

    # compare which is higher between pred and threshold
    check_conf = json.dumps(
        {'Confidence is higher than threshold? ->': check_confidence(prob[pred_order[0]], threshold)})

    pred = json.dumps({'Pred num': pred_order[0]})  # == torch.argmax(pred, 1).item()
    pred_order = json.dumps({'Pred order': pred_order})

    # To make prob more intuitive
    prob = np.round(np.array(prob, np.float32), 6)
    prob = json.dumps({i: str(prob[i]) for i in range(10)}, indent=4)

    print("Confidence is", prob)
    print(pred_order)
    print(pred)
    print(check_conf)

    return pred, pred_order, prob, json.dumps({"threshold": threshold}), check_conf


class Item(BaseModel):
    enc_img: bytes


@app.post("/pred/")
async def return_pred(file: Item):
    return predict(file)
