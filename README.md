# WhisperEncoderWrapper
A simple wrapper for [OpenAI Whisper's](https://github.com/openai/whisper) encoder.

## Setup
I used **Python 3.9.19** and **PyTorch 2.0.1** for testing. You can install this package directly using the following commands, but I strongly recommend that install [PyTorch](https://pytorch.org/) manually to suit your experimental environment. 
```bash
# Install from github repo
pip install git+https://github.com/navi0105/WhisperEncoderWrapper

# If you want to modify this package, you can clone and install it with editable mode
git clone https://github.com/navi0105/WhisperEncoderWrapper
cd WhisperEncoderWrapper
pip install -e .
```

## Example
```python
>>> import torch
>>> from whisper_wrapper.wrapper import WhisperEncoderWrapper
>>> 
>>> encoder = WhisperEncoderWrapper("large-v3", device="cuda")
>>> 
>>> sample_inputs = torch.randn(16, 128, 3000, device="cuda")
>>> with torch.no_grad():
...     out = encoder(sample_inputs)
... 
>>> print(out.shape)
torch.Size([16, 1500, 1280])
```
