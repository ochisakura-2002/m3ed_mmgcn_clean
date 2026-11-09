"""TA-only form of the historical packed UniLSTM and audio projection math."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TAEncoders(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_projection = nn.Linear(config.text_dim, config.encoder_dim)
        self.text_encoder = nn.LSTM(
            config.encoder_dim, config.encoder_dim, num_layers=1,
            batch_first=True, bidirectional=False,
        )
        self.audio_projection = nn.Linear(config.audio_dim, config.encoder_dim)

    def forward(self, text_features, audio_features, lengths, mask):
        projected = self.text_projection(text_features)
        packed = pack_padded_sequence(
            projected, lengths.detach().cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_text, _ = self.text_encoder(packed)
        text, _ = pad_packed_sequence(
            packed_text, batch_first=True, total_length=text_features.shape[1],
        )
        audio = torch.relu(self.audio_projection(audio_features))
        valid = mask.unsqueeze(-1)
        return text * valid, audio * valid
