class Defaults:
    """Default hyperparamter values used in this package's Transformer model."""

    Dropout = 0.1
    EmbeddingSize = 512
    FeedForwardHiddenSize = 2048
    NumberAttentionHeads = 8
    NumberDecoderLayers = 6
    NumberEncoderLayers = 6
    MaxRelativeAttentionDistance = 10
    OptimizerWarmupSteps = 4000

class Hyperparameters:
    def __init__( self,
        vocab_size,
        dropout=Defaults.Dropout,
        embedding_size=Defaults.EmbeddingSize,
        feed_forward_hidden_size=Defaults.FeedForwardHiddenSize,
        number_attention_heads=Defaults.NumberAttentionHeads,
        number_decoder_layers=Defaults.NumberDecoderLayers,
        number_encoder_layers=Defaults.NumberEncoderLayers,
        max_relative_attention_distance=Defaults.MaxRelativeAttentionDistance,
        optimizer_warmup_steps=Defaults.OptimizerWarmupSteps ):
        """Creates a new package of hyperparameters for the Music Transformer model."""
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.feed_forward_hidden_size = feed_forward_hidden_size
        self.number_attention_heads = number_attention_heads
        self.number_decoder_layers = number_decoder_layers
        self.number_encoder_layers = number_encoder_layers
        self.max_relative_attention_distance = max_relative_attention_distance
        self.optimizer_warmup_steps = optimizer_warmup_steps
