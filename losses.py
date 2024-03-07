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

  def pacmap_near_loss(self, neighbor_pairs: Tensor) -> float:
    distance: Tensor = self.pacmap_distance(self, neighbor_pairs)
    ratio: Tensor = distance / (10 + distance)
    loss: float = torch.sum(ratio)
    return loss

  def pacmap_midnear_loss(self, midnear_pairs: Tensor) -> float:
    distance: Tensor = self.pacmap_distance(self, midnear_pairs)
    ratio: Tensor = distance / (10000 + distance)
    loss: float = torch.sum(ratio)
    return loss

  def pacmap_far_loss(self, far_pairs: Tensor) -> float:
    distance: Tensor = self.pacmap_distance(self, far_pairs)
    ratio: Tensor = 1 / (1 + distance)
    loss: float = torch.sum(ratio)
    return loss

  def phase_1(self, t1, t2) -> tuple[float, float, float]:
    neighbor_weight: float = 2.0
    midnear_weight: float = 1000.0 * (1 - ((t1-1)/(t2-1))) + 3 * ((t1-1)/(t2-1))
    far_weight: float = 1.0
    return (neighbor_weight, midnear_weight, far_weight)

  def phase_2(self) -> tuple[float, float, float]:
    neighbor_weight: float = 3.0
    midnear_weight: float = 3.0
    far_weight: float = 1.0
    return (neighbor_weight, midnear_weight, far_weight)

  def phase_3(self) -> tuple[float, float, float]:
    neighbor_weight: float = 1.0
    midnear_weight: float = 0.0
    far_weight: float = 1.0
    return (neighbor_weight, midnear_weight, far_weight)
  
  def __load_pairs(self, type: str, n_pairs, graph: dict, input: Tensor) -> Tensor:
    pairs: Tensor[float] = torch.zeros((n_pairs, 2, input.size[1]))
    
    slot_index: int = 0
    for point_index in graph[type]:
      for paired_index in graph[type][point_index]:
        pairs[slot_index] = [input[paired_index], input[point_index]]
        slot_index += 1

    return pairs

  def forward(self, input: Tensor, n_neighbors: int, n_mn: int, n_fp: int, input_indicies, graph: dict, iteration: int, epochs: int, phase: int):
    neighbor_weight: float; midnear_weight: float; far_weight: float

    match phase:
      case 1:
        neighbor_weight, midnear_weight, far_weight  = self.phase_1(epochs, epochs)
      case 2: 
        neighbor_weight, midnear_weight, far_weight  = self.phase_2()
      case 3:
        neighbor_weight, midnear_weight, far_weight  = self.phase_3()

    n_near_pairs: int = input.size[0] * n_neighbors
    n_mid_pairs: int = input.size[0] * n_mn
    n_far_pairs: int = input.size[0] * n_fp
    
    near_pairs: Tensor[float] = self.__load_pairs('neighbor_pairs', n_near_pairs, graph, input)
    mid_pairs: Tensor[float] = self.__load_pairs('midnear_pairs', n_mid_pairs, graph, input)
    far_pairs: Tensor[float] = self.__load_pairs('far_pairs', n_far_pairs, graph, input)


    near_loss: float = self.pacmap_near_loss(near_pairs, input.size[1]) * neighbor_weight
    midnear_loss: float = self.pacmap_midnear_loss(mid_pairs, input.size[1]) * midnear_weight
    far_loss: float = self.pacmap_far_loss(far_pairs, input.size[1]) * far_weight

    loss = near_loss, midnear_loss, far_loss
    return loss