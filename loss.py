import numpy as np
import torch
from torch import Tensor

class PaCMAPLoss(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def l2_norm(point_1: Tensor, point_2: Tensor, squared=False) -> Tensor:
    x = point_1 - point_2
    x = torch.pow(x, 2)
    x = torch.sum(x)
    if squared == False:
      x = torch.sqrt(x)
    return x

  def pacmap_distance(self, pairs: Tensor) -> Tensor:
    distance: Tensor = torch.empty(pairs.size(dim=0))
    for index, pair in enumerate(pairs):
      distance[index] = self.l2_norm(pair[0], pair[1], squared=True) + 1
    return distance

  def pacmap_near_loss(self, neighbor_pairs: Tensor, dim: int) -> float:
    distance: Tensor = self.pacmap_distance(self, neighbor_pairs)
    ratio: Tensor = distance / (10 + distance)
    loss: float = torch.sum(ratio)
    return loss

  def pacmap_midnear_loss(self, midnear_pairs: Tensor, dim: int) -> float:
    distance: Tensor = self.pacmap_distance(self, midnear_pairs)
    ratio: Tensor = distance / (10000 + distance)
    loss: float = torch.sum(ratio)
    return loss

  def pacmap_far_loss(self, far_pairs: Tensor, dim: int) -> float:
    distance: Tensor = self.pacmap_distance(self, far_pairs)
    ratio: Tensor = 1 / (1 + distance)
    loss: float = torch.sum(ratio)
    return loss

  def phase_1(t1, t2) -> tuple[float, float, float]:
    neighbor_weight: float = 2
    midnear_weight: float = 1000 * (1 - ((t1-1)/(t2-1))) + 3 * ((t1-1)/(t2-1))
    far_weight: float = 1
    return (neighbor_weight, midnear_weight, far_weight)

  def phase_2() -> tuple[float, float, float]:
    neighbor_weight: float = 3
    midnear_weight: float = 3
    far_weight: float = 1
    return (neighbor_weight, midnear_weight, far_weight)

  def phase_3() -> tuple[float, float, float]:
    neighbor_weight: float = 1
    midnear_weight: float = 0
    far_weight: float = 1
    return (neighbor_weight, midnear_weight, far_weight)

  def forward(self, input: Tensor, graph: dict[Tensor], iteration, phase_1=0, phase_2=101, phase_3=201):
    if iteration < phase_2:
      neighbor_weight, midnear_weight, far_weight  = self.phase_1(phase_1, phase_2)
    elif iteration < phase_3:
      neighbor_weight, midnear_weight, far_weight  = self.phase_2()
    else:
      neighbor_weight, midnear_weight, far_weight  = self.phase_3()

    #near_pairs = get_knn(k=10)

    near_loss = self.pacmap_near_loss() * neighbor_weight
    midnear_loss = self.pacmap_midnear_loss() * midnear_weight
    far_loss = self.pacmap_far_loss() * far_weight

    loss = near_loss + midnear_loss + far_loss
    return loss