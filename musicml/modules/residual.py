import torch.nn as nn

from ..hyperp import Defaults

class ResidualNorm( nn.Module ):
    """Residual connection and normalization applied after each sublayer in the encoder and decoder.

    This layer implements a residual connection that sums the input and output of the preceding
    sublayer together. The summed output is then processed with layer normalization. During
    training, a dropout is applied to the sublayer output before it is summed with the sublayer
    input.
    """

    def __init__( self, embedding_size=Defaults.EmbeddingSize, dropout=Defaults.Dropout ):
        """Creates a new residual+norm module.

        Args:
            embedding_size: Dimensionality of the embedding vector space.
            dropout: Dropout rate to use when applied to sublayer output during the forward pass.
        """
        self.dropout = nn.Dropout( dropout )
        self.norm = nn.LayerNorm( embedding_size )

    def forward( self, sublayer_input, sublayer_output ):
        return self.norm( sublayer_input + self.dropout( sublayer_output ) )
