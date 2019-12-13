import numpy as np
import torch
import torch.nn as nn

from .modules.decoder import DecoderStack
from .modules.embedding import Embedding
from .modules.encoder import EncoderStack
from .modules.output import Output

class MusicTransformer( nn.Module ):
    """Module implementing a Music Transformer from the paper by Huang et al. entitled "Music
    Transformer: Generating Music with Long-Term Structure"."""

    def __init__( self, hyper ):
        super().__init__()
        self.input_embedding = Embedding( hyper.vocab_size, hyper.embedding_size )
        self.encoder = EncoderStack( hyper.number_encoder_layers,
            hyper.embedding_size,
            hyper.attention_key_size,
            hyper.attention_value_size,
            hyper.cache_attention_weights )
        self.output = Output( hyper.vocab_size, hyper.embedding_size )

    def forward( self, input_sequence, attention_mask=None ):
        """Runs one pass of the Music Transformer across the given input and output sequences.

        Args:
            input_sequence: Sequence of input tokens to pass through the Music Transformer's
                encoder. Must be a 1D tensor. Length is denoted by S.
            output_sequence: Sequence of output tokens currently generated to pass through the
                Music Transformer's decoder. Must be a 1D tensor. Length is denoted by T.
            attention_mask: A T x S tensor containing the attention mask to apply during the
                decoder's self attention phase. Entries corresponding to values that should be
                masked must be set to -inf, and all other entries must be set to zero.
        """
        # Encode the input source sequence if given. Otherwise use the previously generated results.
        # Embed the input token sequences into the embedded vector space.
        source = self.input_embedding( input_sequence )
        encoder_output = self.encoder( source, attention_mask )

        # Build the final list of scores for each possible output token.
        return self.output( encoder_output )

def create_attention_mask( output_length, input_length ):
    """Create an attention mask that is output_length x input_length.

    The returned matrix will consist of zeros on and below the main diagonal and -inf everywhere
    else.

    Args:
        output_length: Number of tokens in the output sequence.
        input_length: Number of tokens in the input sequence.
    """
    # Adapted from: https://gist.github.com/kolloldas/a7fd453152c5335019f45c96351e2497#file-main-py
    return torch.from_numpy(
        np.triu(
            np.full( (output_length, input_length), np.NINF ),
            k=1
        )
    ).type( torch.FloatTensor )
