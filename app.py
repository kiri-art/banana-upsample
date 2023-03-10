import base64
from io import BytesIO
import time
import PIL
import json
import os
import cv2
import numpy as np
import torch
import re
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer
from models import upsamplers, face_enhancers
from send import send
from safetensors.torch import save_file, load_file

nets = {
    "RRDBNet": RRDBNet,
    "SRVGGNetCompact": SRVGGNetCompact,
}

device_id = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_id)
print(device)

OPTIMIZE = True if os.environ.get("OPTIMIZE", None) else False


class RealESRGANer2(RealESRGANer):
    def __init__(
        self,
        scale,
        model_path,
        dni_weight=None,
        model=None,
        tile=0,
        tile_pad=10,
        pre_pad=10,
        half=False,
        device=None,
        # gpu_id="None",
    ):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        self.device = device
        loadnet = load_file(model_path, device=device_id)
        model.load_state_dict(loadnet, strict=True)

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()


from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean


class GFPGANer2(GFPGANer):
    def __init__(
        self,
        model_path,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=None,
    ):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        self.device = device

        # initialize the GFP-GAN
        if arch == "clean":
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "bilinear":
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "original":
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "RestoreFormer":
            from gfpgan.archs.restoreformer_arch import RestoreFormer

            self.gfpgan = RestoreFormer()

        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            model_rootpath="gfpgan/weights",
        )

        loadnet = load_file(model_path, device=device_id)
        self.gfpgan.load_state_dict(loadnet, strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    # needed for bananna optimizations
    # global model

    global models
    global face_enhancer

    send(
        "init",
        "start",
        {
            "device": torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else "cpu",
            "hostname": os.getenv("HOSTNAME"),
            # "model_id": MODEL_ID,
            "model_id": "(UPSAMPLE-opt)",
        },
        True,
    )

    models = upsamplers
    for model_key in models:
        print("Init " + model_key)
        model = models[model_key]
        modelModel = nets[model["net"]](**model["initArgs"])
        opt_path = re.sub(r".pth$", ".safetensors", model["path"])
        RealESRGANerToUse = RealESRGANer2
        if not os.path.exists(opt_path):
            if OPTIMIZE:
                print(
                    "Optimizing "
                    + model["path"]
                    + " {:,} bytes".format(os.path.getsize(model["path"]))
                )
                t = time.time()
                upsampler = RealESRGANer(
                    scale=model["netscale"],
                    model_path=model["path"],
                    dni_weight=None,
                    model=modelModel,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=True,
                    device=device,
                )
                # upsampler.to(torch.device("cuda")) -- model init does it already
                print("Load time: {:.2f} s".format(time.time() - t))
                # print(upsampler.model.state_dict())
                # model_scripted = torch.jit.script(upsampler.model)
                # torch.save({"params": upsampler.model.state_dict()}, opt_path)
                save_file(upsampler.model.state_dict(), opt_path)
                print("Optimized: {:,} bytes".format(os.path.getsize(opt_path)))
                os.remove(model["path"])
            else:
                opt_path = model["path"]
                RealESRGANerToUse = RealESRGANer

        t = time.time()
        print("Loading " + model["name"])
        upsampler = RealESRGANerToUse(
            scale=model["netscale"],
            model_path=opt_path,
            dni_weight=None,
            model=modelModel,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=device,
        )

        print("Load time: {:.2f} s".format(time.time() - t))
        print()

        model.update(
            {
                "model": modelModel,
                "upsampler": upsampler,
            }
        )

    # GFPGAN
    print("Init GFPGan")
    model_path = face_enhancers["GFPGAN"]["path"]
    opt_path = re.sub(r".pth$", ".safetensors", face_enhancers["GFPGAN"]["path"])
    GFPGANerToUse = GFPGANer2
    if not os.path.exists(opt_path):
        if OPTIMIZE:
            print(
                "Optimizing "
                + model_path
                + " {:,} bytes".format(os.path.getsize(model_path))
            )
            t = time.time()
            print(model_path)
            face_enhancer = GFPGANer(
                model_path=model_path,
                upscale=4,  # args.outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
                device=device,
            )
            # upsampler.to(torch.device("cuda")) -- model init does it already
            print("Load time: {:.2f} s".format(time.time() - t))
            # print(upsampler.model.state_dict())
            save_file(face_enhancer.gfpgan.state_dict(), opt_path)
            print("Optimized: {:,} bytes".format(os.path.getsize(opt_path)))
            os.remove(model_path)
        else:
            GFPGANerToUse = GFPGANer
            opt_path = face_enhancers["GFPGAN"]["path"]

    t = time.time()
    face_enhancer = GFPGANerToUse(
        model_path=opt_path,
        upscale=4,  # args.outscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
        device=device,
    )
    print("Load time: {:.2f} s".format(time.time() - t))
    print()

    # the models are all pretty small, just GFPGAN is about 5x the next
    # biggest, so let's optimize that.
    # model = face_enhancer

    send("init", "done")


def decodeBase64Image(imageStr: str) -> PIL.Image:
    return PIL.Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))


def truncateInputs(inputs: dict):
    clone = inputs.copy()
    if "modelInputs" in clone:
        modelInputs = clone["modelInputs"] = clone["modelInputs"].copy()
        for item in ["input_image"]:
            if item in modelInputs:
                modelInputs[item] = modelInputs[item][0:6] + "..."
    return clone


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(all_inputs: dict) -> dict:
    # global model
    global models

    # use optimized version
    global face_enhancer
    # face_enhancer = model

    print(json.dumps(truncateInputs(all_inputs), indent=2))
    model_inputs = all_inputs.get("modelInputs", None)
    call_inputs = all_inputs.get("callInputs", None)
    startRequestId = call_inputs.get("startRequestId", None)

    model_id = call_inputs.get("MODEL_ID")
    if models.get(model_id, "None") == None:
        return {
            "$error": {
                "code": "MISSING_MODEL",
                "message": f'Model "{model_id}" not available on this container.',
                "requested": model_id,
                # "available": MODEL_ID,
            }
        }

    upsampler = models[model_id]["upsampler"]

    face_enhance = model_inputs.get("face_enhance", False)
    if face_enhance:  # Use GFPGAN for face enhancement
        face_enhancer.bg_upsampler = upsampler

    # TODO... download this model too, switch as needed
    # https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth
    if model_id == "realesr-general-x4v3":
        denoise_strength = model_inputs.get("denoise_strength", 1)
        if denoise_strength != 1:
            # wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            # model_path = [model_path, wdn_model_path]
            # upsampler = models["realesr-general-x4v3-denoise"]
            # upsampler.dni_weight = dni_weight
            dni_weight = [denoise_strength, 1 - denoise_strength]
            return "TODO: denoise_strength"

    if "input_image" not in model_inputs:
        return {
            "$error": {
                "code": "NO_INPUT_IMAGE",
                "message": "Missing required parameter `input_image`",
            }
        }

    # image = decodeBase64Image(model_inputs.get("input_image"))
    image_str = base64.b64decode(model_inputs["input_image"])
    image_np = np.frombuffer(image_str, dtype=np.uint8)
    # bytes = BytesIO(base64.decodebytes(bytes(model_inputs["input_image"], "utf-8")))
    img = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)

    send("inference", "start", {"startRequestId": startRequestId}, True)

    # Run the model
    # with autocast("cuda"):
    #    image = pipeline(**model_inputs).images[0]
    if face_enhance:
        _, _, output = face_enhancer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )
    else:
        output, _rgb = upsampler.enhance(img, outscale=4)  # TODO outscale param

    image_base64 = base64.b64encode(cv2.imencode(".jpg", output)[1]).decode()

    send("inference", "done", {"startRequestId": startRequestId})

    # Return the results as a dictionary
    return {"image_base64": image_base64}


if __name__ == "__main__":
    # optimize models
    init()
