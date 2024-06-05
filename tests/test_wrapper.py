import torch
from whisper_encoder_wrapper.wrapper import WhisperEncoderWrapper

def test_load_wrapper():
    encoder = WhisperEncoderWrapper('base', 'cpu')
    
def test_forward():
    encoder = WhisperEncoderWrapper('base', 'cpu')
    
    x = torch.randn(16, 80, 3000)
    with torch.no_grad():
        y = encoder(x)

    assert y.shape == (16, 1500, 512)