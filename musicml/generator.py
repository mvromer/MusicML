import mido
import torch

from .hyperp import Hyperparameters
from .model import MusicTransformer, create_attention_mask
from .train import pianoecomp as midimodel

class MusicGenerator:
    """Generates music using a trained Music Transformer model."""

    def __init__( self, hyper, weights_path ):
        self.model = MusicTransformer( hyper )
        self.model.load_state_dict( torch.load( weights_path, map_location="cpu" ) )

        self.using_gpu = False
        if torch.cuda.is_available():
            self.model.cuda()
            self.using_gpu = True

    def generate( self, input_sequence, start_token, stop_token, max_output_length=1000 ):
        with torch.no_grad():
            input_sequence = torch.tensor( input_sequence, dtype=torch.long )
            output_sequence = torch.empty( max_output_length, dtype=torch.long )
            output_sequence[0] = start_token

            if torch.cuda.is_available():
                input_sequence = input_sequence.cuda()
                output_sequence = output_sequence.cuda()

            # Run the encoder once for all decode steps.
            self.model( input_sequence=input_sequence )

            # Run the decoder until either a stop token is generated or we've hit our max length.
            output_length = 1
            for next_output_idx in range( 1, output_sequence.size( 0 ) ):
                current_output_sequence = output_sequence[:next_output_idx]
                current_output_length = current_output_sequence.size( 0 )
                attention_mask = create_attention_mask( current_output_length, current_output_length )

                if torch.cuda.is_available():
                    attention_mask = attention_mask.cuda()

                model_output = self.model( output_sequence=current_output_sequence,
                    attention_mask=attention_mask )
                next_output = model_output[-1:, :].argmax().item()
                output_sequence[next_output_idx] = next_output
                output_length += 1

                if next_output == stop_token:
                    break

        return output_sequence[:output_length].tolist()

def generate_from_midi( input_midi_path, output_path, weights_path ):
    hyper = Hyperparameters( len( midimodel.Vocabulary ) )
    generator = MusicGenerator( hyper, weights_path )
    input_sequence = [
        midimodel.VocabularyIndexMap[input_token]
        for input_token in midimodel.convert_to_midi_model( input_midi_path )
    ]
    output_sequence = generator.generate( input_sequence, midimodel.StartTokenIndex, midimodel.StopTokenIndex )
    output_tokens = [
        midimodel.Vocabulary[output_token_idx]
        for output_token_idx in output_sequence
        if output_token_idx != midimodel.StartTokenIndex and output_token_idx != midimodel.StopTokenIndex
    ]

    with open( output_path, "w" ) as output_file:
        output_file.writelines( "\n".join( output_tokens ) )
