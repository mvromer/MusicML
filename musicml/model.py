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

        encoder_layer = nn.TransformerEncoderLayer( d_model=hyper.embedding_size,
            nhead=hyper.number_attention_heads,
            dim_feedforward=hyper.feed_forward_hidden_size,
            dropout=hyper.dropout )
        self.encoder = nn.TransformerEncoder( encoder_layer, hyper.number_encoder_layers )

        self.output = Output( hyper.vocab_size, hyper.embedding_size )
        self.encoder_output = None
        self.encoder_only = hyper.encoder_only

    def forward( self, source_sequence=None, target_sequence=None, source_mask=None, target_mask=None, encode_only=False ):
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
        if source_sequence is not None:
            source = self.input_embedding( source_sequence )

            # NOTE: Pytorch's Transformer implementation requires input to be S x N x E where S is
            # the sequence length, N is the batch size, and E is the embedding size. We don't
            # currently implement batching (or at least attempt to support it), so we'll reshape our
            # 2D S x E source sequence to a S x 1 x E through an unsqueeze. After running that
            # through the encoder, we'll squeeze it back to S x E.
            source = source.unsqueeze( 1 )
            self.encoder_output = self.encoder( source, source_mask ).squeeze( 1 )

            if self.encoder_only:
                return self.output( self.encoder_output )

        if encode_only:
            return self.encoder_output

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
