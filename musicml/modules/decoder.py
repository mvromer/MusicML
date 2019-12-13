import torch.nn as nn

from .attention import MultiheadAttention
from .feedforward import FeedForward
from ..hyperp import Defaults
from .residual import ResidualNorm

class DecoderLayer( nn.Module ):
    """Single layer within the decoder.

    A decoder layer has the following stages or sublayers:

        * Self-attention against the target sequence with relative positional embeddings.
        * Residual connections followed by layer normalization.
        * Encoder-decoder attention using the target sequence and encoder output.
        * Residual connections followed by layer normalization.
        * Feed-forward network.
        * Residual connections followed by layer normalization.
    """

    def __init__( self,
        embedding_size=Defaults.EmbeddingSize,
        attention_key_size=Defaults.AttentionKeySize,
        attention_value_size=Defaults.AttentionValueSize,
        cache_attention_weights=Defaults.CacheAttentionWeights,
        dropout=Defaults.Dropout ):
        super().__init__()

        # First sublayer is masked self attention against the target sequence.
        self.self_attention = MultiheadAttention( embedding_size,
            key_size=attention_key_size,
            value_size=attention_value_size,
            embed_relative_positions=False )
        self.self_attention_norm = nn.LayerNorm( embedding_size )
        self.self_attention_dropout = nn.Dropout( dropout )

        # Second sublayer is encoder-decoder attention against the encodder output and target
        # sequence.
        self.enc_dec_attention = MultiheadAttention( embedding_size,
            key_size=attention_key_size,
            value_size=attention_value_size )
        self.enc_dec_attention_residual = ResidualNorm( embedding_size )
        self.enc_dec_attention_norm = nn.LayerNorm( embedding_size )
        self.enc_dec_attention_dropout = nn.Dropout( dropout )

        # The final sublayer is the feed-forward network.
        self.feed_forward = FeedForward( embedding_size )
        self.feed_formward_norm = nn.LayerNorm( embedding_size )
        self.feed_forward_dropout = nn.Dropout( dropout )

    def forward( self, target, encoder_output, attention_mask ):
        target_norm = self.self_attention_norm( target )
        self_attention_output = self.self_attention( target_norm, target_norm, attention_mask )
        self_attention_output = self.self_attention_dropout( self_attention_output )
        self_attention_output += target

        self_attention_output_norm = self.enc_dec_attention_norm( self_attention_output )
        enc_dec_attention_output = self.enc_dec_attention( encoder_output, self_attention_output_norm )
        enc_dec_attention_output = self.enc_dec_attention_dropout( enc_dec_attention_output )
        enc_dec_attention_output += self_attention_output

        enc_dec_attention_output_norm = self.feed_formward_norm( enc_dec_attention_output )
        feed_forward_output = self.feed_forward( enc_dec_attention_output_norm )
        feed_forward_output = self.feed_forward_dropout( feed_forward_output )
        return enc_dec_attention_output + feed_forward_output

class DecoderStack( nn.Module ):
    """Stack of decoder layers executed in series."""

    def __init__( self,
        number_layers=Defaults.NumberDecoderLayers,
        embedding_size=Defaults.EmbeddingSize,
        attention_key_size=Defaults.AttentionKeySize,
        attention_value_size=Defaults.AttentionValueSize,
        cache_attention_weights=Defaults.CacheAttentionWeights ):
        """Creates a new decoder stack.

        Args:
            number_layers: Number of decoder layers in the decoder stack.
            embedding_size: Dimensionality of each embedding vector in the target sequence. Denoted
                as d(E).
        """
        super().__init__()
        self.decoder_layers = nn.ModuleList( [
            DecoderLayer( embedding_size, attention_key_size, attention_value_size, cache_attention_weights )
            for _ in range( number_layers )
        ] )

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
