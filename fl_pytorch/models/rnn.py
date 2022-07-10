#!/usr/bin/env python3

"""
Reccurent Neural Network for Shakespeare Dataset
"""

# Import PyTorch root package import torch
import torch

import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, vocab_size=90, embedding_dim=8, hidden_dim=512, num_layers=2):
        super(RNN, self).__init__()

        # set class variables
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_lstm_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        embeds = self.embedding(input)
        lstm_out, hidden = self.lstm1(embeds, hidden)
        out = self.fc(lstm_out)
        # flatten the output
        out = out.reshape(-1, self.vocab_size)
        return out, hidden

    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim).to(device = device),
                  torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim).to(device = device))
        return hidden


# TODO: Any way we can actually have an useful pretrained argument here?
def rnn(pretrained=False, num_classes=90):
    return RNN(vocab_size=num_classes)


def minirnn(pretrained=False, num_classes=90):
    return RNN(vocab_size=num_classes, hidden_dim=128)
