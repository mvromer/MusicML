import torch.nn as nn

from .hyperp import Defaults

class Embedding( nn.Module ):
    """Learnable embedding layer for converting sequences of input/output tokens into a dense vector space."""

    def __init__( self, vocab_size, embedding_size=Defaults.EmbeddingSize ):
        super().__init__()
        self.embedding = nn.Embedding( vocab_size, embedding_size )
        self.scale_factor = embedding_size ** -0.5

    def forward( self, x ):
        return scale_factor * self.embedding( x )
