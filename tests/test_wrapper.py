import torch

from whisper_wrapper.wrapper import WhisperEncoderWrapper
from whisper_wrapper.wrapper_hf import WhisperEncoderWrapperHF


def test_load_wrapper():
    encoder = WhisperEncoderWrapper("base", "cpu")


def test_load_wrapper_hf():
    encoder = WhisperEncoderWrapperHF.from_pretrained("openai/whisper-base")


def test_forward():
    encoder_openai = WhisperEncoderWrapper("base", "cpu")
    encoder_hf = WhisperEncoderWrapperHF.from_pretrained("openai/whisper-base")

    x = torch.randn(16, 80, 3000)
    with torch.no_grad():
        y_openai = encoder_openai(x)
        y_hf = encoder_hf(x).last_hidden_state

    assert y_openai.shape == (16, 1500, 512)
    assert y_hf.shape == (16, 1500, 512)

    assert torch.equal(y_openai, y_hf)
