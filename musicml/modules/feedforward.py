import torch.nn as nn

from .hyperp import Defaults

class FeedForward( nn.Module ):
    """Feed-forward network applied after the multihead attention sublayers within each layer of the
    decoder and encoder."""

    def __init__( self,
        embedding_size=Defaults.EmbeddingSize,
        hidden_size=Defaults.FeedForwardHiddenSize ):
        """Creates a new feed-forward network to use in either the encoder or decoder.

        Args:
            embedding_size: Dimensionality of the embedding vector space.
            hidden_size: Dimensionality of the hidden layer in the feed-forward network. The default
            is 2048, which is the same used in the paper Attention Is All You Need.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear( embedding_size, hidden_size ),
            nn.ReLU(),
            nn.Linear( hidden_size, embedding_size )
        )

    def forward( self, x ):
        return self.network( x )
