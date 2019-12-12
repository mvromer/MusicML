import torch.nn as nn

from .attention import MultiheadAttention
from ..hyperp import Defaults
from .feedforward import FeedForward
from .residual import ResidualNorm

class EncoderLayer( nn.Module ):
    """Single layer within the encoder.

    An encoder layer has the following stages or sublayers:

        * Self-attention against the source sequence with relative positional embeddings.
        * Residual connections followed by layer normalization.
        * Feed-forward network.
        * Another set of residual connections followed by layer normalization.
    """

    def __init__( self,
        embedding_size=Defaults.EmbeddingSize,
        attention_key_size=Defaults.AttentionKeySize,
        attention_value_size=Defaults.AttentionValueSize,
        dropout=Defaults.Dropout ):
        super().__init__()

        # First sublayer is self attention with the source sequence as input.
        self.self_attention = MultiheadAttention( embedding_size,
            key_size=attention_key_size,
            value_size=attention_value_size,
            embed_relative_positions=False )
        self.self_attention_norm = nn.LayerNorm( embedding_size )
        self.self_attention_dropout = nn.Dropout( dropout )

        # Second sublayer is the feed-forward network.
        self.feed_forward = FeedForward( embedding_size )
        self.feed_formward_norm = nn.LayerNorm( embedding_size )
        self.feed_forward_dropout = nn.Dropout( dropout )

    def forward( self, source ):
        # NOTE: There appears to be some discrepancy between what the Vaswani paper says for the
        # ordering of the sublayer with respect to the residual connection and layer normalization
        # and what is actually implemented in the authors' reference implementation. This uses the
        # order prescribed in the reference implementation, which normalizes the input the sublayer,
        # applies a dropout to the sublayer output, and then adds the residual connection. This also
        # applies to the decoder layer.
        source_norm = self.self_attention_norm( source )
        self_attention_output = self.self_attention( source_norm, source_norm )
        self_attention_output = self.self_attention_dropout( self_attention_output )
        self_attention_output += source

        self_attention_output_norm = self.feed_formward_norm( self_attention_output )
        feed_forward_output = self.feed_forward( self_attention_output_norm )
        feed_forward_output = self.feed_forward_dropout( feed_forward_output )
        return self_attention_output + feed_forward_output

class EncoderStack( nn.Module ):
    """Stack of encoder layers executed in series."""

    def __init__( self,
        number_layers=Defaults.NumberEncoderLayers,
        embedding_size=Defaults.EmbeddingSize,
        attention_key_size=Defaults.AttentionKeySize,
        attention_value_size=Defaults.AttentionValueSize ):
        """Creates a new encoder stack.

        Args:
            number_layers: Number of encoder layers in the encoder stack.
            embedding_size: Dimensionality of each embedding vector in the source sequence. Denoted
                as d(E).
        """
        super().__init__()
        self.encoder_layers = nn.ModuleList( [
            EncoderLayer( embedding_size, attention_key_size, attention_value_size )
            for _ in range( number_layers )
        ] )

    def forward( self, source ):
        """Passes the given source sequence through the encoder stack.

        Args:
            source: Source sequence of embedding vectors. Has dimensions S x d(E).
        """
        x = source
        for layer in self.encoder_layers:
            x = layer( x )
        return x
