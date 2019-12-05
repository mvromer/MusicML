import torch.nn as nn

from .modules.decoder import DecoderStack
from .modules.embedding import Embedding
from .modules.encoder import EncoderStack
from .modules.output import Output

class MusicTranformer( nn.Module ):
    """Module implementing a Music Transformer from the paper by Huang et al. entitled "Music
    Transformer: Generating Music with Long-Term Structure"."""

    def __init__( self, hyper ):
        self.input_embedding = Embedding( hyper.vocab_size, hyper.embedding_size )
        self.output_embedding = Embedding( hyper.vocab_size, hyper.embedding_size )
        self.encoder = EncoderStack( hyper.number_encoder_layers, hyper.embedding_size )
        self.decoder = DecoderStack( hyper.number_decoder_layers, hyper.embedding_size )
        self.output = Output( hyper.vocab_size, hyper.embedding_size )

    def forward( self, input_sequence, output_sequence, attention_mask ):
        # Embed the input and output token sequences into the embedded vector space.
        source = self.input_embedding( input_sequence )
        target = self.output_embedding( output_sequence )

        # Encode the source.
        encoder_output = self.encoder( source )

        # Decode the next output in the target.
        decoder_output = self.decoder( target, encoder_output, attention_mask )

        # Build the final list of probabilities for each possible output token.
        return self.output( decoder_output )
