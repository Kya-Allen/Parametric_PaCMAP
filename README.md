#Parametric PaCMAP. (this project is incomplete and not yet in a functional condition)

A Parametric reformulation of the Pairwise Controlled Manifold Approximation Projection (PaCMAP) algorithm by Wang et al., Inspired by Sainburg et al's parametric reformulation of the Uniform Manifold Approximation Projection (UMAP) algorithm.

Unlike prior dimensionality reduction algorithms such as UMAP and t-sne, PaCMAP stands out and above the rest by employing a multistage embedding process that applies differential losses for different kinds of point pairs beased on their nearness, where each stage alters the weights on the losses of these kinds of pairs. This highly controlled pairwise loss allows PaCMAP to excel at preserving both the local and global distances in the data.

This project consists of two parts, both built on PyTorch:
  * a new PyTorch Loss class "PaCMAPLoss(torch.nn.Module)" which implements the pairwise controlled logic of the PaCMAP loss, but suited for training a neural network, and progressing through stages as the network progresses through Epcohs
  * a simple [insert final number] layer encoder network "ParametricPaCMAP(torch.nn.Module). (of course it should be easy to use your own custom encoder).

In the future, look forward to performance analyses in the same vein as Seinburg et al., with parametric UMAP


@article{JMLR:v22:20-1061,
  author  = {Yingfan Wang and Haiyang Huang and Cynthia Rudin and Yaron Shaposhnik},
  title   = {Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {201},
  pages   = {1-73},
  url     = {http://jmlr.org/papers/v22/20-1061.html}
}

@article{sainburg2021parametric,
  title={Parametric UMAP embeddings for representation and semisupervised learning},
  author={Sainburg, Tim and McInnes, Leland and Gentner, Timothy Q},
  journal={Neural Computation},
  volume={33},
  number={11},
  pages={2881--2907},
  year={2021},
  publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
}
