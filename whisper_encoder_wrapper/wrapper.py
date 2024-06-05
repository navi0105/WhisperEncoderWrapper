import os
import hashlib
import warnings
import urllib
from dataclasses import dataclass
from typing import Union, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import whisper
from whisper.model import AudioEncoder, ModelDimensions

_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

# default model dimensions settings, based on whisper base model
DEFAULT_DIMS = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=512,
    n_audio_head=64,
    n_audio_layer=6,
    n_vocab=51865,
    n_text_ctx=448,
    n_text_state=512,
    n_text_head=8,
    n_text_layer=6,
)

def _download(url: str, root: str) -> Union[bytes, str]:
    """
        Function for downloading checkpoint files of pretrained Whisper model, modified from OpenAI's Whisper library ().
    """
    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return download_target


def _load_from_whisper(
        model_name: str,
        device: str="cuda",
    ) -> Tuple[dict, ModelDimensions]:

    default = os.path.join(os.path.expanduser("~"), ".cache")
    download_root = os.path.join(
        os.getenv("XDG_CACHE_HOME", default), 
        "whisper"
    )

    checkpoint_file = _download(
        _MODELS[model_name], 
        download_root, 
    )

    with open(checkpoint_file, "rb") as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    encoder_state_dict = {}
    for k, v in checkpoint["model_state_dict"].items():
        if k.startswith("encoder."):
            key = k.replace("encoder.", "")
            encoder_state_dict[key] = v

    del checkpoint
    return encoder_state_dict, dims

def _load_from_checkpoint(
        model_path: str, 
        device: str="cuda"
    ) -> Tuple[dict, ModelDimensions]:
    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"
    checkpoint = torch.load(model_path, map_location=device)

    dims = ModelDimensions(**checkpoint["dims"])
    encoder_state_dict = checkpoint["encoder_state_dict"]

    del checkpoint
    return encoder_state_dict, dims
    

class WhisperEncoderWrapper(nn.Module):
    def __init__(
        self, 
        model_name_or_path: str=None,
        device: str="cuda"
    ):
        super().__init__()

        self.dims = DEFAULT_DIMS
        self.device = device

        if model_name_or_path is not None:
            if model_name_or_path in _MODELS:
                encoder_state_dict, dims = _load_from_whisper(model_name_or_path, device)
            else:
                encoder_state_dict, dims = _load_from_checkpoint(model_name_or_path, device)

            self.dims = dims
        else:
            encoder_state_dict = None

        self.encoder = AudioEncoder(
            n_mels=self.dims.n_mels,
            n_ctx=self.dims.n_audio_ctx,
            n_state=self.dims.n_audio_state,
            n_head=self.dims.n_audio_head,
            n_layer=self.dims.n_audio_layer,
        )
        if encoder_state_dict is not None:
            self.encoder.load_state_dict(encoder_state_dict)

        self.encoder = self.encoder.to(device)

    def save_model(self, path: str):
        state_dict = {}
        state_dict["dims"] = self.dims.__dict__
        state_dict["encoder_state_dict"] = self.encoder.state_dict()

        torch.save(state_dict, path)

    def forward(self, x):
        return self.encoder(x)