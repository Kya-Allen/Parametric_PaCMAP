import numpy as np
import torch
from torch import Tensor

class PaCMAPLoss(torch.nn.Module):
  def __init__(self):
    super().__init__()

  # Tested works great
  def pacmap_distance(self, pairs: Tensor) -> Tensor:
    norms = torch.norm(pairs[:, 0] - pairs[:, 1], dim=1, p=2, keepdim=False)
    return norms.pow(2) + 1

  def pacmap_near_loss(self, neighbor_pairs: Tensor, dim: int) -> float:
    distance : Tensor = self.pacmap_distance(self, neighbor_pairs)
    ratio : Tensor = distance / (10 + distance)
    loss : float = torch.sum(ratio)
    return loss

  def pacmap_midnear_loss(self, midnear_pairs: Tensor, dim: int) -> float:
    distance : Tensor = self.pacmap_distance(self, midnear_pairs)
    ratio : Tensor = distance / (10000 + distance)
    loss : float = torch.sum(ratio)
    return loss

  def pacmap_far_loss(self, far_pairs: Tensor, dim: int) -> float:
    distance : Tensor = self.pacmap_distance(self, far_pairs)
    ratio : Tensor = 1 / (1 + distance)
    loss : float = torch.sum(ratio)
    return loss

  def phase_1(self, t1, t2) -> tuple[float, float, float]:
    neighbor_weight : float = 2
    midnear_weight : float = 1000 * (1 - ((t1-1)/(t2-1))) + 3 * ((t1-1)/(t2-1))
    far_weight: float = 1
    return (neighbor_weight, midnear_weight, far_weight)

  def phase_2(self) -> tuple[float, float, float]:
    neighbor_weight : float = 3
    midnear_weight : float = 3
    far_weight: float = 1
    return (neighbor_weight, midnear_weight, far_weight)

  def phase_3(self) -> tuple[float, float, float]:
    neighbor_weight : float = 1
    midnear_weight : float = 0
    far_weight: float = 1

  def forward(self, input: Tensor, graph: dict[Tensor], iteration, phase_1, phase_2, phase_3):
    if iteration < phase_2:
      neighbor_weight, midnear_weight, far_weight  = self.phase_1(phase_1, phase_2)
    elif iteration < phase_3:
      neighbor_weight, midnear_weight, far_weight  = self.phase_2()
    else:
      neighbor_weight, midnear_weight, far_weight  = self.phase_3()

    near_loss = self.pacmap_near_loss() * neighbor_weight
    midnear_loss = self.pacmap_midnear_loss() * midnear_weight
    far_loss = self.pacmap_far_loss() * far_weight

    loss = near_loss, midnear_loss, far_loss
    return loss