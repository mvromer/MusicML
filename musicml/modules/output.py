import torch.nn as nn

from ..hyperp import Defaults

class Output( nn.Module ):
    """Output layer used to convert the results of the model into the target output space.

    The result of this layer is a 2D tensor shaped T x V where T is the length of the target
    sequence and V is the size of the output vocabulary. Each row in the resulting tensor contains a
    set of raw scores computed across all possible tokens in the output vocabulary. In particular,
    the each score in the final row of the resulting tensor relates to the probability that the
    corresponding token is next output token produced by the model.
    """

    def __init__( self, vocab_size, embedding_size=Defaults.EmbeddingSize ):
        """Creates a new output layer.

        Args:
            embedding_size: Dimensionality of the embedding vector space.
            vocab_size: Dimensionality of the target output space.
        """
        super().__init__()

        # The output layer is simply a linear transformation from the embedding vector space to the
        # output vocabulary space.
        self.output = nn.Linear( embedding_size, vocab_size, bias=False )

    def forward( self, x ):
        return self.output( x )
