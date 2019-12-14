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
        embed_relative_positions=Defaults.EmbedRelativePositions,
        cache_attention_weights=Defaults.CacheAttentionWeights,
        dropout=Defaults.Dropout ):
        super().__init__()

        # First sublayer is self attention with the source sequence as input.
        self.self_attention = MultiheadAttention( embedding_size,
            key_size=attention_key_size,
            value_size=attention_value_size,
            embed_relative_positions=embed_relative_positions,
            cache_attention_weights=cache_attention_weights )
        self.self_attention_residual = ResidualNorm( embedding_size )

        # Second sublayer is the feed-forward network.
        self.feed_forward = FeedForward( embedding_size )
        self.feed_forward_residual = ResidualNorm( embedding_size )

    def forward( self, source, attention_mask=None ):
        # NOTE: There appears to be some discrepancy between what the Vaswani paper says for the
        # ordering of the sublayer with respect to the residual connection and layer normalization
        # and what is actually implemented in the authors' reference implementation. This uses the
        # order prescribed in the reference implementation, which normalizes the input the sublayer,
        # applies a dropout to the sublayer output, and then adds the residual connection. This also
        # applies to the decoder layer.
        self_attention_output = self.self_attention( source, source, attention_mask )
        self_attention_output = self.self_attention_residual( source, self_attention_output )

        feed_forward_output = self.feed_forward( self_attention_output )
        return self.feed_forward_residual( self_attention_output, feed_forward_output )

class EncoderStack( nn.Module ):
    """Stack of encoder layers executed in series."""

    def __init__( self,
        number_layers=Defaults.NumberEncoderLayers,
        embedding_size=Defaults.EmbeddingSize,
        attention_key_size=Defaults.AttentionKeySize,
        attention_value_size=Defaults.AttentionValueSize,
        embed_relative_positions=Defaults.EmbedRelativePositions,
        cache_attention_weights=Defaults.CacheAttentionWeights ):
        """Creates a new encoder stack.

        Args:
            number_layers: Number of encoder layers in the encoder stack.
            embedding_size: Dimensionality of each embedding vector in the source sequence. Denoted
                as d(E).
        """
        super().__init__()
        self.encoder_layers = nn.ModuleList( [
            EncoderLayer( embedding_size,
                attention_key_size,
                attention_value_size,
                embed_relative_positions,
                cache_attention_weights )
            for _ in range( number_layers )
        ] )

    def forward( self, source, attention_mask=None ):
        """Passes the given source sequence through the encoder stack.

        Args:
            source: Source sequence of embedding vectors. Has dimensions S x d(E).
        """
        x = source
        for layer in self.encoder_layers:
            x = layer( x, attention_mask )
        return x
