# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import os
import requests
from models import models_by_type


def download_model():
    os.makedirs("weights", exist_ok=True)
    for type in models_by_type:
        models = models_by_type[type]
        for model_key in models:
            model = models[model_key]
            print(model)
            response = requests.get(model["weights"])
            open(model["path"], "wb").write(response.content)

    os.makedirs("/gfpgan/weights")
    response = requests.get(
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
    )
    open("/gfpgan/weights/detection_Resnet50_Final.pth", "wb").write(response.content)
    response = requests.get(
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
    )
    open("/gfpgan/weights/parsing_parsenet.pth", "wb").write(response.content)


if __name__ == "__main__":
    download_model()
