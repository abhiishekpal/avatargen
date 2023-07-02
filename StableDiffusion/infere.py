import requests
from PIL import Image
import io
import base64

payload = {
    "prompt": "random person  <lora:Abhishek:1> riding a motorbike",
    "steps": 5
}


response = requests.post(url=f'http://127.0.0.1:7861/sdapi/v1/txt2img', json=payload)
r = response.json()


for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
    im1 = image.save(r"C:\Users\91973\Documents\AIGRAM\test.png")