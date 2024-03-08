import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike, NDArray
import torch
from torch import Tensor, float64, int64
from torch.optim import Optimizer
import losses
from sklearn.neighbors import NearestNeighbors

def pacmap_distance(pair: Tensor) -> Tensor:
  norms = torch.linalg.vector_norm(pair[0] - pair[1], ord=2)
  return norms.pow(2) + 1

class PaCMAPEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim=3):
      super().__init__()
      self.input_dim = input_dim
      if output_dim > 24: raise Exception('output_dim must be 24 or less')
      self.encoder = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 192), # our project is 384 input
        torch.nn.ReLU(),
        torch.nn.Linear(192, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, output_dim),
      )


    def forward(self, data: Tensor):
      valid = self.validate_input(data)
      if not valid[0]: raise Exception(valid[1])
      return self.encoder(data)

    def validate_input(self, data) -> tuple[bool, str]:
      return
      
class PaCMAPRefiner(torch.nn.Module):
    def __init__(self, input_dim):
      super().__init__()
      self.input_dim = input_dim
      self.net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, input_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(input_dim, input_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(input_dim, input_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(input_dim, input_dim),
      )


    def forward(self, data: Tensor):
      valid = self.validate_input(data)
      if not valid[0]: raise Exception(valid[1])
      return self.net(data)

    def validate_input(self, data) -> tuple[bool, str]:
      return


class ParametricPacMAP():
  def __init__(self, data: Tensor, output_dim: int, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0):
    self.data: Tensor = data
    self.size: int = data.shape[0]
    if output_dim > 24: raise Exception('output_dim must be 24 or less')
    self.output_dim: int = output_dim
    self.dimensionality: int = data.shape[1]
    self.n_neighbors: int = n_neighbors
    self.mn_ratio: float = mn_ratio
    self.fp_ratio: float = fp_ratio
    self.output_data = Tensor([])

  graph: dict = {
        #adjacency graph for each type of edge
        'neighbor_pairs': {},
        'midnear_pairs': {},
        'far_pairs': {}
        # note, batch can be achieved by constructing a seperate graph per batch.
        # other parameters like n_neighbors may need tuning.
    }

  def train_batchless(self, epochs: int, optimizer_type='Adam'):
    self.networks: tuple[torch.nn.Module] = (PaCMAPEncoder(self.dimensionality, self.output_dim), PaCMAPRefiner(self.dimensionality), PaCMAPRefiner(self.dimensionality))
    optimizers: Optimizer = self.__set_optimizers(optimizer_type)
    if self.graph['neighbor_pairs'] == {}:
      self.compute_graph()
    for epoch in range(epochs):
        input: Tensor = self.data
        for net_index, network in enumerate(self.networks):
          # Forward pass
          output: Tensor = network.forward(input, self.graph, self.n_neighbors, self.mn_ratio, self.fp_ratio)
          loss: Tensor = loss.PaCMAPLoss(output, self.n_neighbors, self.n_mn, self.n_fp, self.neighbor_graph, epoch, epochs, phase=net_index)

          # backward and optimize
          optimizers[net_index].zero_grad()
          loss.backward()
          optimizers[net_index].step()

          # set output as input for next network
          input = output

  def __set_optimizers(self, optimizer_type) -> tuple[Optimizer]:
    match optimizer_type:
      case 'Adam':
        optimizers = (
          torch.optim.Adam(self.networks[0]),
          torch.optim.Adam(self.networks[1]),
          torch.optim.Adam(self.networks[2])
              )
      case 'SGD':
       optimizers = (
          torch.optim.SGD(self.networks[0]),
          torch.optim.SGD(self.networks[1]),
          torch.optim.SGD(self.networks[2])
              )
    return optimizers

  def compute_graph(self) -> dict:
    for index, datum in enumerate(self.data):
      self.graph['neighbor_pairs'][index] = self.__get_neighbors(datum).type(int64)
      self.graph['midnear_pairs'][index] = self.__get_midnears(datum).type(int64)
      self.graph['far_pairs'][index] = self.__get_farpairs().type(int64)
    return self.graph

  def __get_neighbors(self, datum: Tensor) -> Tensor:
    initial_set_size: int = min(self.n_neighbors + 50, self.dimensionality)
    neighborhood: NearestNeighbors = NearestNeighbors(n_neighbors=initial_set_size)
    neighborhood.fit(self.data)
    nearest_distances, nearest_indices = neighborhood.kneighbors(datum, return_distance=True)
    neighbors: list[tuple] = self.__pacmap_nearest_neighbors(datum, self.data, nearest_indices.flatten(), self.n_neighbors)
    return Tensor([neighbor[1] for neighbor in neighbors])

  def __get_midnears(self, query: Tensor) -> Tensor:
    self.n_mn: int =  int(self.n_neighbors * self.mn_ratio) #is this right?
    midnears: Tensor = torch.zeros(self.n_mn)
    for neighbor_slot in midnears:
      sample_map: ndarray[int] = np.random.randint(self.size, size=6)
      neighbor: tuple = self.__pacmap_nearest_neighbors(query, self.data, sample_map, 2)[1]
      neighbor_slot = neighbor[1]
    return midnears 
  
  def __get_farpairs(self) -> Tensor:
    self.n_fp: int = int(self.n_neighbors * self.fp_ratio)
    sampled_indicies: Tensor[int] = torch.randint(0, self.size, (self.n_fp,))
    return sampled_indicies
  
  def __pacmap_nearest_neighbors(self, query: Tensor, data: Tensor, nearest_indices: ndarray, n_neighbors: int) -> list[tuple[float, int]]:
    nearest_indices: Tensor = torch.from_numpy(nearest_indices)
    pacmap_distances: Tensor = torch.zeros(nearest_indices.size()[0])
    for loop_index, neighbor_index in enumerate(nearest_indices):
      pair: Tensor = torch.zeros((2, self.dimensionality), dtype=float64)
      pair[0] = query;
      pair[1] = data[neighbor_index]
      pacmap_distances[loop_index] = pacmap_distance(pair)

    neighborhood: zip = zip(pacmap_distances, nearest_indices)
    neighborhood: list[tuple] = sorted(neighborhood, key=lambda x: x[0])
    return neighborhood[0:n_neighbors]