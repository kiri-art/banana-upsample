# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
import os
from io import BytesIO
from PIL import Image

TESTS = "tests"
FIXTURES = TESTS + os.sep + "fixtures"
OUTPUT = TESTS + os.sep + "output"


def b64encode_file(filename: str):
    with open(os.path.join(FIXTURES, filename), "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def output_path(filename: str):
    return os.path.join(OUTPUT, filename)


def test(name, json):
    print("Running test: " + name)
    res = requests.post("http://localhost:8000/", json=json)
    json = res.json()
    print(json)

    image_byte_string = json["image_base64"]

    image_encoded = image_byte_string.encode("utf-8")
    image_bytes = BytesIO(base64.b64decode(image_encoded))
    image = Image.open(image_bytes)
    fp = output_path(name + ".jpg")
    image.save(fp)
    print("Saved " + fp)
    print()


test(
    "RealESRGAN_x4plus_anime_6B",
    {
        "modelInputs": {
            "input_image": b64encode_file("Anime_Girl.svg.png"),
        },
        "callInputs": {"MODEL_ID": "RealESRGAN_x4plus_anime_6B"},
    },
)

test(
    "RealESRGAN_x4plus",
    {
        "modelInputs": {
            "input_image": b64encode_file("Blake_Lively.jpg"),
            "face_enhance": True,
        },
        "callInputs": {"MODEL_ID": "RealESRGAN_x4plus"},
    },
)
