import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike, NDArray
import torch
from torch import Tensor
import losses
from sklearn.neighbors import NearestNeighbors

def pacmap_distance(pairs: Tensor) -> Tensor:
  norms = torch.norm(pairs[:, 0] - pairs[:, 1], dim=1, p=2, keepdim=False)
  return norms.pow(2) + 1

class PaCMAPNet(torch.nn.Module):
    def __init__(self, input_dim):
      super().__init__()
      self.input_dim = input_dim
      self.encoder = torch.nn.Sequential(
        torch.nn.Linear(384, 192),
        torch.nn.ReLU(),
        torch.nn.Linear(192, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, 3),
      )


    def forward(self, data: Tensor, n_neighbors, mn_ratio, FP_ratio):
      if self.nn_graph == {}:
        self.compute_graph()
      valid = self.validate_input(data)
      if not valid[0]: raise Exception(valid[1])
      return self.encoder(data)

    def validate_input(self, data) -> tuple[bool, str]:
      return


class ParametricPacMAP():
  def __init__(self, data: Tensor, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0):
    self.data: Tensor = data
    self.size: int = data.shape[0]
    self.dimensionality: int = data.shape[1]
    self.n_neighbors: int = n_neighbors
    self.mn_ratio: float = mn_ratio
    self.fp_ratio: float = fp_ratio

    self.network = PaCMAPNet(self.dimensionality)

  graph: dict = {
        #adjacency graph for each type of edge
        'neighbor_pairs': {},
        'midnear_pairs': {},
        'far_pairs': {}
    }

  def train(self, epochs: int, optimizer: torch.Optimizer, phase_1=0, phase_2=100, phase_3=200):
    if self.graph['neighbor_pairs'] == {}:
      self.compute_graph()
    for epoch in range(epochs):
      for datum in self.data:
        # Forward pass
        output: Tensor = self.network(self.data, self.n_neighbors, self.mn_ratio, self.FP_ratio)
        loss = loss.PaCMAPLoss(output, self.neighbor_graph, epoch, phase_1, phase_2, phase_3)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  def compute_graph(self) -> dict:
    for index, datum in enumerate(self.data):
      self.graph['neighbor_pairs'][index] = self.__get_neighbors(datum)
      self.graph['midnear_pairs'][index] = self.__get_midnears(datum)
      self.graph['far_pairs'][index] = self.__get_farpairs(datum)
      return self.nn_graph

  def __get_neighbors(self, datum: Tensor[float]) -> Tensor[int]:
    initial_set_size: int = min(self.n_neighbors + 50, self.dimensionality)
    neighborhood: NearestNeighbors = NearestNeighbors(initial_set_size)
    neighborhood.fit(self.data)
    nearest_distances, nearest_indices = neighborhood.kneighbors(datum, return_distance=True)
    neighbors: list[tuple] = self.__pacmap_nearest_neighbors(datum, self.data, nearest_distances, nearest_indices, self.n_neighbors)
    return Tensor([neighbor[1] for neighbor in neighbors])

  def __get_midnears(self, query: Tensor[float]) -> Tensor[int]:
    n_midnears: int =  int(self.n_neighbors * self.mn_ratio) #is this right?
    midnears: Tensor = torch.zeros((1, n_midnears))
    for neighbor_slot in midnears:
      sample_map: ndarray[int] = np.random.randint(self.size, size=6)
      neighbor: tuple = self.__pacmap_nearest_neighbors(query, self.data, sample_map, 2)[1]
      midnears[neighbor_slot] = neighbor[1]
    return midnears 
  
  def __get_farpairs(self) -> Tensor[int]:
    n_farpairs: int = int(self.n_neighbors * self.fp_ratio)
    sampled_indicies: Tensor[int] = torch.randint(0, self.size, (n_farpairs,))
    return sampled_indicies
  
  def __pacmap_nearest_neighbors(self, query: Tensor[float], data: Tensor, nearest_indices: ndarray, n_neighbors: int) -> list[tuple[float, int]]:
    nearest_indices: Tensor = torch.from_numpy(nearest_indices)
    pacmap_distances: Tensor = torch.zeros((1, len(nearest_indices)))
    for loop_index, neighbor_index in enumerate(nearest_indices):
      pair: Tensor = Tensor([query, data[neighbor_index]])
      pacmap_distances[loop_index] = pacmap_distance(pair)

    neighborhood: zip = zip(pacmap_distances, nearest_indices)
    neighborhood: list[tuple] = sorted(neighborhood, key=lambda x: x[0])
    return neighborhood[0:n_neighbors]