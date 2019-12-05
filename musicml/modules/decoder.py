import torch.nn as nn

from .attention import MultiheadAttention
from .feedforward import FeedForward
from .hyperp import Defaults
from .residual import ResidualNorm

class DecoderLayer( nn.Module ):
    """Single layer within the decoder.

    A decoder layer has the following stages or sublayers:

        * Self-attention against the target sequence.
        * Residual connections followed by layer normalization.
        * Encoder-decoder attention using the target sequence and encoder output.
        * Residual connections followed by layer normalization.
        * Feed-forward network.
        * Residual connections followed by layer normalization.
    """

    def __init__( self, embedding_size=Defaults.EmbeddingSize ):
        super().__init__()
        # First sublayer is masked self attention against the target sequence.
        self.self_attention = MultiheadAttention( embedding_size )
        self.self_attention_residual = ResidualNorm( embedding_size )

        # Second sublayer is encoder-decoder attention against the encodder output and target
        # sequence.
        self.enc_dec_attention = MultiheadAttention( embedding_size )
        self.enc_dec_attention_residual = ResidualNorm( embedding_size )

        # The final sublayer is the feed-forward network.
        self.feed_forward = FeedForward( embedding_size )
        self.feed_forward_residual = ResidualNorm( embedding_size )

    def forward( self, target, encoder_output, attention_mask ):
        x = self.self_attention( target, target, attention_mask )
        x = self.self_attention_residual( x )
        x = self.enc_dec_attention( encoder_output, target )
        x = self.enc_dec_attention_residual( x )
        x = self.feed_forward( x )
        return self.feed_forward_residual( x )

class DecoderStack( nn.Module ):
    """Stack of decoder layers executed in series."""

    def __init__( self,
        number_layers=Defaults.NumberDecoderLayers,
        embedding_size=Defaults.EmbeddingSize ):
        """Creates a new decoder stack.

        Args:
            number_layers: Number of decoder layers in the decoder stack.
            embedding_size: Dimensionality of each embedding vector in the target sequence. Denoted
                as d(E).
        """
        super().__init__()
        self.decoder_layers = [
            DecoderLayer( embedding_size )
            for _ in range( number_layers )
        ]

    def forward( self, target, encoder_output, attention_mask ):
        """Passes the given target sequence, encoder outputs, and attention mask through the decoder
        stack.

        Args:
            target: Target sequence of embedding vectors. Has dimensions T x d(E).
            encoder_output: Output of the encoder stack. Has dimensions S x d(E).
            attention_mask: Additive mask to apply when computing self attention within the decoder.
                Has dimensions T x S.
        """
        x = target
        for layer in self.decoder_layers:
            x = layer( target, encoder_output, attention_mask )
        return x
