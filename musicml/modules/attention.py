import torch
import torch.nn as nn

from .hyperp import Defaults

class MultiheadAttention( nn.Module ):
    """Multihead attention mechanism based on the one described in Attention Is All You Need from Vaswani et al.

    Notes:
        When called, this will take in two tensors: a source sequence whose dimensions are S x E and
        a target sequence whose dimensions are T x E. Keys and values are projected from the source
        sequence, and queries are projected from the target sequence.

        This layer will use scaled dot-product attention across multiple heads. The results are
        projected back to an output tensor with dimensions T x d(E).
    """
    def __init__( self, kv_length,
        embedding_size=Defaults.EmbeddingSize, number_heads=Defaults.NumberAttentionHeads,
        query_length=None, key_size=None, value_size=None ):
        """Creates a new multihead attention layer.

        Args:
            kv_length: Number of key-value pairs projected by each head. Denoted as S.
            embedding_size: Dimensionality of the source/target sequence's embedding vectors.
                Denoted as d(E).
            number_heads: Number of heads attending. Denoted as H.
            query_length: Number of queries projected by each head. Denoted as T. If not given,
                defaults to kv_length.
            key_size: Dimensionality of each key projected. Denoted as d(K). If not given, defaults
                to embedding_size.
            value_size: Dimensionality of each value projected. Denoted as d(V). If not given,
                defaults to embedding_size.
        """
        super().__init__()
        self.kv_length = kv_length
        self.embedding_size = embedding_size
        self.number_heads = number_heads
        self.query_length = query_length or kv_length
        self.key_size = key_size or embedding_size
        self.value_size = value_size or embedding_size

        # Transformations that project source sequences into key and value spaces. Expressed as a
        # linear layer with no bias.
        self.key_trans = nn.Linear( self.embedding_size, self.key_size * self.number_heads, bias=False )
        self.value_trans = nn.Linear( self.embedding_size, self.value_size * self.number_heads, bias=False )

        #  Transformation that projects target sequences into query space. Expressed as a linear
        # layer with no bias.
        self.query_trans = nn.Linear( self.embedding_size, self.key_size * self.number_heads, bias=False )

        # Scaling factor applied after computing QK^T.
        self.scale_factor = self.key_size ** -0.5

        # Softmax layer applied to innermost dimension of Z = (QK^T) / sqrt(d(K)) during attention
        # calculation.
        self.z_softmax = nn.Softmax( dim=-1 )

        # Projection from the matrix of attention values concatenated from each head to our original
        # embedding vector space. Expressed as linear layer with no bias.
        self.z_trans = nn.Linear( self.value_size * self.number_heads, self.embedding_size, bias=False )

    def forward( self, source, target ):
        # Project keys, values, and queries from inputs.
        keys = self.key_trans( source )
        values = self.value_trans( source )
        queries = self.query_trans( target )

        # Each output can be viewed as a 1-by-H block matrix containing the keys, queries, and
        # values for each head in the multihead attention mechanism. Each block is sized as follows:
        #
        #   For keys: S x d(K)
        #   For values: S x d(V)
        #   For queries: T x d(K)
        #
        # To perform a batch multiplication, we need to view each as a block tensor with the
        # following size:
        #
        #   For keys: H x d(K) x S (because we need the transpose of each block)
        #   For values: H x S x d(V)
        #   For queries: H x T x d(K)
        #
        # For values and queries, we need transpose the linear layer output, view it as a 3D tensor,
        # and then permute the dimensions. For the keys (since we need the transpose), we only need
        # to tranpose the linear layer output and view it as a 3D tensor.
        #
        keyTransView = keys.t().view( self.number_heads, self.key_size, self.kv_length )
        queriesView = queries.t().view( self.number_heads, self.kv_length, self.query_length ).permute( 0, 2, 1 )
        valuesView = values.t().view( self.number_heads, self.value_size, self.kv_length ).permute( 0, 2, 1 )

        # Start computing attention in parallel across all heads. First Z = QK^T.
        z = torch.bmm( queriesView, keyTransView )

        # Scale by root inverse of K.
        z = self.scale_factor * z

        # Softmax along innermost dimension of Z.
        z = self.z_softmax( z )

        # Finally multiply by V to get final attention value for each head.
        z = torch.bmm( z, valuesView )

        # Concatenate each head's attention value horizontally across Z to make it T x (d(V) * H).
        z = torch.cat( torch.unbind( z, dim=0 ), dim=1 )

        # Project the final multihead attention value back into our original embedding vector space.
        z = z_trans( z )
        return z
