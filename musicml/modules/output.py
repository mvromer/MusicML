import torch.nn as nn

class Output( nn.Module ):
    """Output layer used to convert the results of the model into the target output space.

    The result of this layer is a vector whose dimensionality is the size of the output vocabulary
    space. Each element of the resulting vector is the probability the corresponding token is the
    next output token produced by the model.
    """

    def __init__( self, embedding_size, vocab_size ):
        """Creates a new output layer.

        Args:
            embedding_size: Dimensionality of the embedding vector space.
            vocab_size: Dimensionality of the target output space.
        """
        super().__init__()

        # The output layer consists of two sublayers: a linear transformation from the embedding
        # vector space to the output vocabulary space and a softmax to convert the outputs to
        # probabilities.
        self.output = nn.Sequential(
            nn.Linear( embedding_size, vocab_size, bias=False ),
            nn.Softmax()
        )

    def forward( self, x ):
        return self.output( x )
