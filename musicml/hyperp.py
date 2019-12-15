class Defaults:
    """Default hyperparamter values used in this package's Transformer model."""

    EncoderOnly = True
    Dropout = 0.1
    EmbeddingSize = 512
    FeedForwardHiddenSize = 2048
    NumberAttentionHeads = 8

    # Based on query/key hidden size of 512 from Huang et al.
    # AttentionKeySize is 512 / NumberAttenionHeads
    AttentionKeySize = 64

    # Based on value hidden size of 256 from Huang et al.
    # AttentionValueSize is 256 / NumberAttenionHeads
    AttentionValueSize = 32

    NumberDecoderLayers = 6
    NumberEncoderLayers = 6

    MaxRelativeAttentionDistance = 500
    EmbedRelativePositions = True
    CacheAttentionWeights = False

    OptimizerWarmupSteps = 4000
    LabelSmoothingValue = 0.1

class Hyperparameters:
    def __init__( self,
        vocab_size,
        encoder_only=Defaults.EncoderOnly,
        dropout=Defaults.Dropout,
        feed_forward_hidden_size=Defaults.FeedForwardHiddenSize,
        number_attention_heads=Defaults.NumberAttentionHeads,
        attention_key_size=Defaults.AttentionKeySize,
        attention_value_size=Defaults.AttentionValueSize,
        number_decoder_layers=Defaults.NumberDecoderLayers,
        number_encoder_layers=Defaults.NumberEncoderLayers,
        max_relative_attention_distance=Defaults.MaxRelativeAttentionDistance,
        embed_relative_positions=Defaults.EmbedRelativePositions,
        cache_attention_weights=Defaults.CacheAttentionWeights,
        optimizer_warmup_steps=Defaults.OptimizerWarmupSteps ):
        """Creates a new package of hyperparameters for the Music Transformer model."""
        self.vocab_size = vocab_size
        self.embedding_size = vocab_size
        self.encoder_only = encoder_only
        self.dropout = dropout
        self.feed_forward_hidden_size = feed_forward_hidden_size
        self.number_attention_heads = number_attention_heads
        self.attention_key_size = attention_key_size
        self.attention_value_size = attention_value_size
        self.number_decoder_layers = number_decoder_layers
        self.number_encoder_layers = number_encoder_layers
        self.max_relative_attention_distance = max_relative_attention_distance
        self.embed_relative_positions = embed_relative_positions
        self.cache_attention_weights = cache_attention_weights
        self.optimizer_warmup_steps = optimizer_warmup_steps
