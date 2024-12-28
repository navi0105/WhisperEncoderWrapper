import torch.nn as nn
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoder,
    WhisperPreTrainedModel,
)


class WhisperEncoderWrapperHF(WhisperPreTrainedModel):
    """ """

    def __init__(self, config):
        super().__init__(config)
        config.is_encoder_decoder = False
        self.encoder = WhisperEncoder(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.encoder.conv1 = value

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


def main():
    pass


if __name__ == "__main__":
    main()
