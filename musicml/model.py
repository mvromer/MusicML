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
        self.output_embedding = Embedding( hyper.vocab_size, hyper.embedding_size )
        self.encoder = EncoderStack( hyper.number_encoder_layers, hyper.embedding_size )
        self.decoder = DecoderStack( hyper.number_decoder_layers, hyper.embedding_size )
        self.output = Output( hyper.vocab_size, hyper.embedding_size )

    def forward( self, input_sequence, output_sequence, attention_mask ):
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
        # Embed the input and output token sequences into the embedded vector space.
        source = self.input_embedding( input_sequence )
        target = self.output_embedding( output_sequence )

        # Encode the source.
        encoder_output = self.encoder( source )

        # Decode the next output in the target.
        decoder_output = self.decoder( target, encoder_output, attention_mask )

        # Build the final list of probabilities for each possible output token.
        return self.output( decoder_output )
