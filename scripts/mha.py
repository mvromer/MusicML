import torch
import torch.nn as nn

# Example of implementing multihead attention sublayer inside a Transformer. This in particular is
# modeling encoder-decoder attention, although the core logic is the same for self attention; the
# only difference is the inputs fed into the algorithm.

# S - Source sequence length, a.k.a, sequence length of encoder output (i.e., decoder's memory).
# T - Target sequence length, a.k.a, sequence length of decoder input.
# E - Dimensionality of input embeddings.
# H - Number of heads in multihead attention.
# K - Dimensionality of the key space in multihead attention.
# V - Dimensionality of the value space in multihead attention.
S = 2
T = 3
E = 5

H = 2

K = 4
V = 2

# Inputs: Memory M and decoder sequence D.
M = torch.rand( (S, E) )
D = torch.rand( (T, E) )

# Key/Value/Query projections expressed as linear layers. We don't want any bias terms here.
key_trans = nn.Linear( E, K * H, bias=False )
query_trans = nn.Linear( E, K * H, bias=False )
value_trans = nn.Linear( E, V * H, bias=False )

# Softmax layer applied to innermost dimension of Z = (QK^T) / sqrt(K) during attention calculation.
z_softmax = nn.Softmax( dim=-1 )

# Projection from multihead attention values back into our model's feature space, expressed as a
# linear layer with no bias term.
z_trans = nn.Linear( V * H, E, bias=False )

# Project keys, values, and queries from inputs.
keys = key_trans( M )
queries = query_trans( D )
values = value_trans( M )

# Each output can be viewed as a 1-by-H block matrix containing the keys, queries, and values for
# each head in the multihead attention mechanism. Each block is sized as follows:
#
#   For keys: S x K
#   For values: S x V
#   For queries: T x K
#
# To perform a batch multiplication, we need to view each as a block tensor with the following size:
#
#   For keys: H x K x S (because we need the transpose of each block)
#   For values: H x S x V
#   For queries: H x T x K
#
# For values and queries, we need transpose the linear layer output, view it as a 3D tensor, and
# then permute the dimensions. For the keys (since we need the transpose), we only need to tranpose
# the linear layer output and view it as a 3D tensor.
#
keyTransView = keys.t().view( H, K, S )
queriesView = queries.t().view( H, K, T ).permute( 0, 2, 1 )
valuesView = values.t().view( H, V, S ).permute( 0, 2, 1 )

# Start computing attention in parallel across all heads. First Z = QK^T.
z = torch.bmm( queriesView, keyTransView )

# Scale by root inverse of K.
scale_factor = K ** -0.5
z = scale_factor * z

# Softmax along innermost dimension of Z.
z = z_softmax( z )

# Finally multiply by V to get final attention value for each head.
z = torch.bmm( z, valuesView )

# Concatenate each head's attention value horizontally across Z to make it T x VH.
z = torch.cat( torch.unbind( z, dim=0 ), dim=1 )

# Project the final multihead attention value back into our model's feature space.
z = z_trans( z )
