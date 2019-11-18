import requests
import json
import base64
import click
# from pathlib import Path


@click.command()
@click.argument('path')
def interact(path):
    # path = Path(path)

    with open(path, 'rb') as f:
        # f.read() is bytes type
        # 1. For security, we need to encode by base64 (still bytes type)
        # 2. FastAPI requires json: bytes -> str (json cannot dump bytes type)
        enc_img = base64.b64encode(f.read()).decode('utf-8')
        body = json.dumps({'enc_img': enc_img})
        r = requests.post(url, data=body)

    print(r.text)


if __name__ == '__main__':
    url = "http://127.0.0.1:8000/pred/"
    interact()
