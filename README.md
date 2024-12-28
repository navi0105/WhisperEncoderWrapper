# WhisperEncoderWrapper

## Introduction
A simple wrapper for [OpenAI Whisper's](https://github.com/openai/whisper) encoder.
I also referred to the `WhisperDecoderWrapper` module in the `transformers` library to implement `WhisperEncoderWrapperHF` module based on the Huggingface version Whisper model. It is basically the same as the OpenAi version, so you can choose the one that suits your needs.

## Setup
I used **Python 3.9.19** and **PyTorch 2.0.1** for testing. You can install this package directly using the following commands, but I strongly recommend installing [PyTorch](https://pytorch.org/) manually to match your experimental environment.

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
>>> from whisper_wrapper.wrapper_hf import WhisperEncoderWrapperHF
>>>
>>> encoder_openai = WhisperEncoderWrapper('large-v3', device="cpu")
>>> encoder_hf = WhisperEncoderWrapperHF.from_pretrained('openai/whisper-large-v3')
>>>
>>> x = torch.randn(16, 128, 3000)
>>> with torch.no_grad():
...     out_openai = encoder_openai(x)
...     out_hf = encoder_hf(x).last_hidden_state
...
>>> out_openai.shape
torch.Size([16, 1500, 1280])
>>> out_hf.shape
torch.Size([16, 1500, 1280])
>>> torch.equal(out_openai, out_hf)
True

```
