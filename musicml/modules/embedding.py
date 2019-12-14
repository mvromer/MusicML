import torch.nn as nn

from ..hyperp import Defaults

class Embedding( nn.Module ):
    """Learnable embedding layer for converting sequences of input/output tokens into a dense vector space."""

    def __init__( self, vocab_size, embedding_size=Defaults.EmbeddingSize ):
        super().__init__()
        self.embedding = nn.Embedding( vocab_size, embedding_size )
        self.positional = PositionalEncoding( embedding_size )
        self.scale_factor = embedding_size ** -0.5

    def forward( self, x ):
        return self.positional( self.scale_factor * self.embedding( x ) )


import math
import torch
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to embedding vectors.

    Adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__( self, d_model, dropout=0.1, max_len=5000 ):
        super().__init__()
        self.dropout = nn.Dropout( p=dropout )

        # Compute the positional encodings once in log space.
        pe = torch.zeros( max_len, d_model )
        position = torch.arange( 0, max_len ).unsqueeze( 1 )
        div_term = torch.exp( torch.arange( 0, d_model, 2 ) * -(math.log( 10000.0 ) / d_model) )
        pe[:, 0::2] = torch.sin( position * div_term )
        pe[:, 1::2] = torch.cos( position * div_term )
        self.register_buffer( 'pe', pe )

    def forward( self, x ):
        x = x + self.pe[:x.size( 0 ), :]
        return self.dropout( x )
