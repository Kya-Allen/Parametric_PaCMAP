import numpy as np
import torch
import loss

class ParametricPaCMAP(torch.nn.Module):
    def __init__(self, input_dim):
      super().__init__()
      self.input_dim = input_dim
      self.encoder = torch.nn.sequential(
        torch.nn.Linear(384, 192),
        torch.nn.ReLU(),
        torch.nn.Linear(192, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, 3),
      )


    def forward(self, data):
      return self.encoder(data)