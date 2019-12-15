import math
import torch
import torch.nn as nn

from ..hyperp import Defaults

class LearnableEmbedding( nn.Module ):
    """Learnable embedding layer for converting sequences of input/output tokens into a dense vector space."""

    def __init__( self, vocab_size, embedding_size=Defaults.EmbeddingSize ):
        super().__init__()
        self.embedding = nn.Embedding( vocab_size, embedding_size )
        self.positional = PositionalEncoding( embedding_size )
        self.scale_factor = embedding_size ** -0.5

    def forward( self, x ):
        return self.positional( self.scale_factor * self.embedding( x ) )

class OneHotEmbedding( nn.Module ):
    """Fixed embedding layer that one-hot encodes each input token."""

    def __init__( self, vocab_size ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained( torch.eye( vocab_size ) )

    def forward( self, x ):
        return self.embedding( x )

class AbsolutePositionalEncoding( nn.Module ):
    """Absolute positional encoding basedd on sinusoids.

    Adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__( self, embedding_size, dropout=0.1, max_length=5000 ):
        super().__init__()
        self.dropout = nn.Dropout( dropout )

        # Compute the positional encodings once in log space.
        positional_encodings = torch.zeros( max_length, embedding_size )
        position = torch.arange( 0, max_length ).unsqueeze( 1 )
        div_term = torch.exp( torch.arange( 0, embedding_size, 2 ) * -(math.log( 10000.0 ) / embedding_size) )
        positional_encodings[:, 0::2] = torch.sin( position * div_term )
        positional_encodings[:, 1::2] = torch.cos( position * div_term )
        self.register_buffer( "positional_encodings", positional_encodings )

    def forward( self, x ):
        x = x + self.positional_encodings[:x.size( 0 ), :]
        return self.dropout( x )
