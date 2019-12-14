import mido
import torch
import torch.nn.functional as F

from .hyperp import Hyperparameters
from .model import MusicTransformer, create_attention_mask
from .train import midimodel

class MusicGenerator:
    """Generates music using a trained Music Transformer model."""

    def __init__( self, hyper, weights_path=None ):
        self.model = MusicTransformer( hyper )
        if weights_path:
            self.model.load_state_dict( torch.load( weights_path, map_location="cpu" ) )
        self.model.eval()

        self.using_gpu = False
        if torch.cuda.is_available():
            self.model.cuda()
            self.using_gpu = True

        self.output_sequence = None

    def generate_sequence( self, priming_sequence, generated_length ):
        for _ in self.generate_outputs( priming_sequence, generated_length ):
            pass
        output_sequence = self.output_sequence.tolist()
        self.output_sequence = None
        return output_sequence

    def generate_outputs( self, priming_sequence, generated_length ):
        # For the generation task, we only used a trained encoder.
        assert self.model.encoder_only

        with torch.no_grad():
            priming_length = len( priming_sequence )
            self.output_sequence = torch.empty( len( priming_sequence ) + generated_length, dtype=torch.long )
            self.output_sequence[:priming_length] = torch.LongTensor( priming_sequence )

            if torch.cuda.is_available():
                self.output_sequence = self.output_sequence.cuda()

            for next_output_offset in range( generated_length ):
                next_output_idx = priming_length + next_output_offset
                current_output_sequence = self.output_sequence[:next_output_idx]
                current_output_length = current_output_sequence.size( 0 )

                print( "-----------" )
                print( f"Next output index: {next_output_idx}" )
                print( f"Current output length: {current_output_length}" )
                print( f"Current output sequence: {current_output_sequence}" )

                model_output = self.model( source_sequence=current_output_sequence )
                next_output_scores = model_output[-1, :]
                next_output_probabilities = F.softmax( next_output_scores, dim=-1 )
                next_output_value = torch.multinomial( next_output_probabilities, 1 ).item()
                self.output_sequence[next_output_idx] = next_output_value
                yield (next_output_value, next_output_scores)

            self.output_sequence = self.output_sequence.cpu()

def generate_from_midi( input_midi_path, output_path, weights_path, **hyper_kwargs ):
    hyper = Hyperparameters( len( midimodel.Vocabulary ), **hyper_kwargs )
    generator = MusicGenerator( hyper, weights_path )
    priming_sequence = [
        midimodel.VocabularyIndexMap[input_token]
        for input_token in midimodel.convert_to_midi_model( input_midi_path )
    ]
    output_sequence = generator.generate_sequence( priming_sequence )
    output_tokens = [
        midimodel.Vocabulary[output_token_idx]
        for output_token_idx in output_sequence
        if output_token_idx != midimodel.StartTokenIndex and output_token_idx != midimodel.StopTokenIndex
    ]

    with open( output_path, "w" ) as output_file:
        output_file.writelines( "\n".join( output_tokens ) )
