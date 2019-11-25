import requests
import json
import base64
import click
from pprint import pprint


@click.command()
@click.argument('path')
@click.argument('threshold')
def interact(path, threshold):
    url = "http://127.0.0.1:8000/pred/"

    with open(path, 'rb') as f:
        # f.read() is bytes type
        # 1. For security, we need to encode by base64 (still bytes type)
        # 2. FastAPI requires json: bytes -> str (json cannot dump bytes type)
        enc_img = base64.b64encode(f.read()).decode('utf-8')
        body = json.dumps({'enc_img': enc_img, 'threshold': threshold})
        r = requests.post(url, data=body).json()

    print('---------------------------------------------------')
    print('pred_num:', r['pred_num'])
    print('pred_order:', r['pred_order'])
    pprint({'prob:': r['prob']})
    print('threshold:', r['threshold'])
    print('model confidence is higher than threshold?:', r['check_conf'])
    print('---------------------------------------------------')


if __name__ == '__main__':
    interact()
